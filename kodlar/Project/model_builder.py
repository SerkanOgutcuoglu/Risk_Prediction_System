# model_builder.py

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os
import pickle

# config dosyasından gerekli sabitleri içe aktar
from config import OUTPUT_DIR, SEQUENCE_LENGTH 

def build_and_train_model(input_shape_rnn, X_train_seq, X_test_seq, y_train_seq_scaled, y_test_seq_scaled):

    print("\n--- Model Oluşturuluyor ve Eğitiliyor ---")
    
    model = Sequential([
        LSTM(64, activation='relu', input_shape=input_shape_rnn, return_sequences=True),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        Dropout(0.3),
        Dense(1) # Regresyon görevi olduğu için çıkış katmanı 1 nöronlu
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()


    # Erken durdurma: val_loss 10 epoch boyunca iyileşmezse eğitimi durdur ve en iyi ağırlıkları geri yükle
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Model kaydetme: En iyi doğrulama kaybına sahip modeli kaydet
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, 'risk_prediction_model.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # Modeli eğit
    history = model.fit(
        X_train_seq, y_train_seq_scaled,
        epochs=100, # Yeterince büyük bir epoch sayısı EarlyStopping durduracak
        batch_size=32,
        validation_data=(X_test_seq, y_test_seq_scaled),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1 # Eğitim ilerlemesini göster
    )



    
    return model, history

def evaluate_model_r2(model, X_test_seq, y_test_seq_scaled, target_scaler):

    print("\n Model Değerlendiriliyor (Test Seti R2 Skoru) ")
    
    # Tahminleri ölçekli formda al
    y_pred_scaled = model.predict(X_test_seq)
    
    # Orijinal ölçeğe geri dönüştür

    y_test_seq_scaled_reshaped = y_test_seq_scaled.reshape(-1, 1) 
    
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled)
    y_test_original = target_scaler.inverse_transform(y_test_seq_scaled_reshaped) 
    
    # R2 skoru hesapla
    r2 = r2_score(y_test_original, y_pred_original)
    print(f"Test Seti R2 Skoru: {r2:.4f}")
    
    # Ortalama Mutlak Hata (MAE)
    loss, mae = model.evaluate(X_test_seq, y_test_seq_scaled, verbose=0)
    print(f"Test Seti MAE (Ölçeklenmiş): {mae:.4f}")
    
    # Orijinal ölçekteki MAE'yi de hesaplayabiliriz
    mae_original = np.mean(np.abs(y_test_original - y_pred_original))
    print(f"Test Seti MAE (Orijinal Ölçek): {mae_original:.4f}")
    return r2


def predict_single_entry(model, combined_df, preprocessor, target_scaler, user_profiles, risk_feature_mappings, numerical_features, categorical_features_for_preprocessing):

    

    print("\n--- Tek Bir Giriş İçin Tahmin Yapılıyor ---")

    if len(combined_df) < SEQUENCE_LENGTH:
        print(f"Uyarı: Tahmin için sağlanan giriş sayısı ({len(combined_df)}) SEQUENCE_LENGTH'ten ({SEQUENCE_LENGTH}) az. Dizinin başı sıfırlarla doldurulacaktır.")
        
    # Ön işlemden geçirilecek sütunlar listesi
    features_to_transform = numerical_features + categorical_features_for_preprocessing
    
    # Sadece dönüştürülecek sütunları içeren geçici bir DataFrame oluştur
    df_for_transform = combined_df[features_to_transform]
    
    # Ön işlenmiş veriyi al 
    processed_data = preprocessor.transform(df_for_transform).toarray()

    # Zaman serisi dizisi oluştur 

    feature_dimension = processed_data.shape[1]
    
    single_sequence = np.zeros((SEQUENCE_LENGTH, feature_dimension))
    actual_sequence_length = len(processed_data)
    
    # Gerçek verileri dizinin sonuna yerleştir
    single_sequence[SEQUENCE_LENGTH - actual_sequence_length:] = processed_data
    
    # Modeli tahmin için hazırla
    single_sequence_reshaped = single_sequence.reshape(1, SEQUENCE_LENGTH, feature_dimension) 
    
    # Tahmin yap
    # model.predict çıktısı (batch_size, 1) şeklindedir, bu yüzden [0][0] ile tek değeri alırız.
    predicted_scaled_score = model.predict(single_sequence_reshaped)[0][0] 
    
    # Tahmin edilen skoru orijinal ölçeğe dönüştür
    predicted_original_score = target_scaler.inverse_transform([[predicted_scaled_score]])[0][0]

    # Gerçek risk skorunu al (combined_df'deki en son girdinin 'RiskScore' sütunundan)

    actual_risk_score = combined_df.iloc[-1]['RiskScore'] 

    predicted_original_score_percent = predicted_original_score * 100
    actual_risk_score_percent = actual_risk_score * 100

    # Output'u .2f formatında (iki ondalık basamak) gösterecek şekilde güncellendi
    print(f"Gerçek Risk Skoru: {actual_risk_score_percent:.2f}") 
    print(f"Tahmin Edilen Risk Skoru: {predicted_original_score_percent:.2f}") 

    # Eşik değeri hala 0-1 aralığında kalmalı (0.50 -> %50)
    if predicted_original_score > 0.50: 
        print("-> Bu giriş Yüksek Riskli olarak değerlendirilmiştir.")
    else:
        print("-> Bu giriş Düşük Riskli olarak değerlendirilmiştir.")
