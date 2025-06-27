# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
import os

from config import SEQUENCE_LENGTH, OUTPUT_DIR

def create_preprocessors(df_columns_for_fit, risk_feature_mappings):
    """
    ColumnTransformer ve hedef ölçekleyiciyi oluşturur ve döndürür.
    Bu fonksiyon sadece preprocessor'ı fit etmek için gerekli sütunları alır.
    """
    # Kategorik ve Sayısal Sütunları Ayıralım
    categorical_features = ['MFAMethod', 'Application', 'Browser', 'OS', 'Unit', 'Title']
    

    numerical_features = [
        'CreatedAt_Hour', 'CreatedAt_DayOfWeek', 'CreatedAt_Month'
    ] + [col_name for col_name in risk_feature_mappings.keys()]


    categorical_features_for_preprocessing = categorical_features + ['ClientIP_Block']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_for_preprocessing)
        ],
        remainder='passthrough' # Diğer sütunları bırak (UserId, CreatedAt, RiskScore, IsRisky_Scenario_Gen)
    )

    # Preprocessor'ı fit etmeden önce tüm ilgili sütunları seçiyoruz.
    preprocessor.fit(df_columns_for_fit[numerical_features + categorical_features_for_preprocessing])
    

    with open(os.path.join(OUTPUT_DIR, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(preprocessor, f)

    # Hedef değişkeni için StandardScaler
    target_scaler = StandardScaler()

    target_scaler.fit(df_columns_for_fit['RiskScore'].values.reshape(-1, 1))

    with open(os.path.join(OUTPUT_DIR, 'target_scaler.pkl'), 'wb') as f:
        pickle.dump(target_scaler, f)
    
    return preprocessor, target_scaler, numerical_features, categorical_features_for_preprocessing

def create_sequences(df, preprocessor, target_scaler, numerical_features, categorical_features_for_preprocessing):

    # preprocessor.transform için doğru sütunları seç
    features_to_transform = numerical_features + categorical_features_for_preprocessing
    
    # Sadece dönüştürülecek sütunları içeren geçici bir DataFrame oluştur
    df_for_transform = df[features_to_transform]

    feature_dimension = preprocessor.transform(df_for_transform.head(1)).shape[1]
    print(f"Her bir zaman adımının özelliği boyutu: {feature_dimension}")

    X_sequences = [] # Modelin girdisi olacak diziler
    y_targets_scaled = []   # Her dizinin hedef skoru

    print("\nZaman serisi dizileri oluşturuluyor...")
    for user_id in df['UserId'].unique():
        user_df = df[df['UserId'] == user_id].sort_values(by='CreatedAt')
        
        # Kullanıcının ön işlenmiş verilerini al

        user_processed_data = preprocessor.transform(user_df[features_to_transform]).toarray()
        
        for i in range(len(user_df)):
            current_entry_index = i
            start_index = max(0, current_entry_index - (SEQUENCE_LENGTH - 1))
            
            sequence_data = np.zeros((SEQUENCE_LENGTH, feature_dimension))
            actual_sequence_length = current_entry_index - start_index + 1
            
            sequence_data[SEQUENCE_LENGTH - actual_sequence_length:] = user_processed_data[start_index : current_entry_index + 1]
            
            X_sequences.append(sequence_data)
            
            # Hedef skoru standardize et
            current_risk_score = user_df.iloc[current_entry_index]['RiskScore']
            y_targets_scaled.append(target_scaler.transform([[current_risk_score]])[0][0])

    X_sequences = np.array(X_sequences)
    y_targets_scaled = np.array(y_targets_scaled)

    print(f"\nOluşturulan toplam zaman serisi dizisi: {X_sequences.shape[0]}")
    print(f"Her dizinin şekli (adım sayısı, özellik boyutu): {X_sequences.shape[1:]}")

    # Eğitim ve Test Kümelerine Ayırma
    X_train_seq, X_test_seq, y_train_seq_scaled, y_test_seq_scaled = train_test_split(
        X_sequences, y_targets_scaled, test_size=0.2, random_state=42
    )

    input_shape_rnn = (X_train_seq.shape[1], X_train_seq.shape[2])
    print(f"Modelin girdi boyutu (zaman adımı, özellik boyutu): {input_shape_rnn}")

    return X_train_seq, X_test_seq, y_train_seq_scaled, y_test_seq_scaled, input_shape_rnn