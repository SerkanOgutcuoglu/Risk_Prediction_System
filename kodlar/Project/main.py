# main.py

import os
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 

# Kendi modüllerimizi içe aktarıyoruz
from config import SEQUENCE_LENGTH, OUTPUT_DIR, RISK_WEIGHTS
from data_generator import generate_mock_data
from feature_engineering import apply_feature_engineering, calculate_risk_score
from preprocessing import create_preprocessors, create_sequences
from model_builder import build_and_train_model, evaluate_model_r2

def train_and_save_all_assets():

    print("Kullanıcı profilleri ve giriş kayıtları oluşturuluyor...")
    df, user_profiles = generate_mock_data()
    print(f"Toplam {len(df)} giriş kaydı ve {len(user_profiles)} kullanıcı profili oluşturuldu.")

    print("\n--- Özellik Mühendisliği ve Kural Tabanlı Risk Etiketleme Başlıyor ---")
    df, risk_feature_mappings = apply_feature_engineering(df.copy(), user_profiles)
    df['RiskScore'] = df.apply(lambda row: calculate_risk_score(row, RISK_WEIGHTS, risk_feature_mappings), axis=1)
    

    df['CreatedAt_Hour'] = df['CreatedAt'].dt.hour
    df['CreatedAt_DayOfWeek'] = df['CreatedAt'].dt.dayofweek
    df['CreatedAt_Month'] = df['CreatedAt'].dt.month
    df['ClientIP_Block'] = df['ClientIP'].apply(lambda x: '.'.join(x.split('.')[:-1]) + '.')

    print("Özellik mühendisliği tamamlandı.")

    print("\n--- Veri Ön İşleme ve Dizi Oluşturma Başlıyor ---")
    preprocessor, target_scaler, numerical_features, categorical_features_for_preprocessing = \
        create_preprocessors(df.copy(), risk_feature_mappings)

    X_train_seq, X_test_seq, y_train_seq_scaled, y_test_seq_scaled, input_shape_rnn = \
        create_sequences(df, preprocessor, target_scaler, numerical_features, categorical_features_for_preprocessing)
    print("Veri ön işleme ve dizi oluşturma tamamlandı.")

    print("\n--- Model Eğitimi Başlıyor ---")
    model, history = build_and_train_model(input_shape_rnn, X_train_seq, X_test_seq, y_train_seq_scaled, y_test_seq_scaled)
    print("Model eğitimi tamamlandı.")

    print("\n--- Model Değerlendirme ---")
    r2 = evaluate_model_r2(model, X_test_seq, y_test_seq_scaled, target_scaler)
    print(f"Model R2 Skoru: {r2:.4f}")

    # Modeli, preprocessor'ı, scaler'ı ve diğer gerekli nesneleri kaydet
    model_path = os.path.join(OUTPUT_DIR, 'risk_prediction_model.h5')
    preprocessor_path = os.path.join(OUTPUT_DIR, 'preprocessor.pkl')
    target_scaler_path = os.path.join(OUTPUT_DIR, 'target_scaler.pkl')
    user_profiles_path = os.path.join(OUTPUT_DIR, 'user_profiles.pkl')
    risk_feature_mappings_path = os.path.join(OUTPUT_DIR, 'risk_feature_mappings.pkl')
    numerical_features_path = os.path.join(OUTPUT_DIR, 'numerical_features.pkl')
    categorical_features_for_preprocessing_path = os.path.join(OUTPUT_DIR, 'categorical_features_for_preprocessing.pkl')
    initial_df_path = os.path.join(OUTPUT_DIR, 'initial_df.pkl') # initial_df'i de kaydet!

    # OUTPUT_DIR'ın var olduğundan emin ol
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Varlıkları kaydet
    print("\n--- Varlıklar Kaydediliyor ---")
    model.save(model_path)
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    with open(target_scaler_path, 'wb') as f:
        pickle.dump(target_scaler, f)
    with open(user_profiles_path, 'wb') as f:
        pickle.dump(user_profiles, f)
    with open(risk_feature_mappings_path, 'wb') as f:
        pickle.dump(risk_feature_mappings, f)
    with open(numerical_features_path, 'wb') as f:
        pickle.dump(numerical_features, f)
    with open(categorical_features_for_preprocessing_path, 'wb') as f:
        pickle.dump(categorical_features_for_preprocessing, f)
    with open(initial_df_path, 'wb') as f:
        pickle.dump(df, f) # Eğitilmiş df'i initial_df olarak kaydet

    print("Model ve tüm ilgili varlıklar başarıyla kaydedildi.")

if __name__ == '__main__':
    train_and_save_all_assets()