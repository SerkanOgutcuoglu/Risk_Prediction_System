# app.py

from flask import Flask, request, jsonify, render_template
from datetime import datetime
import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf

# Kendi modüllerimizi içe aktarıyoruz
# config.py'den gerekli tüm sabitleri içe aktarır
from config import SEQUENCE_LENGTH, OUTPUT_DIR, MFA_METHODS, APPLICATIONS, BROWSERS, OSS, UNITS, TITLES, RISK_WEIGHTS

from feature_engineering import get_risk_feature, calculate_risk_score


app = Flask(__name__)

# Global değişkenler - uygulama başladığında yüklenecekler
model = None
preprocessor = None
target_scaler = None
user_profiles = None
risk_feature_mappings = None
numerical_features = None
categorical_features_for_preprocessing = None
initial_df = None # Model eğitimi için kullanılan başlangıç DataFrame'i, yüklenmeli

def load_all_assets():


    global model, preprocessor, target_scaler, user_profiles, risk_feature_mappings, numerical_features, categorical_features_for_preprocessing, initial_df

    print("Model ve ilgili varlıklar yükleniyor...")

    # Dosya yolları
    model_path = os.path.join(OUTPUT_DIR, 'risk_prediction_model.h5')
    preprocessor_path = os.path.join(OUTPUT_DIR, 'preprocessor.pkl')
    target_scaler_path = os.path.join(OUTPUT_DIR, 'target_scaler.pkl')
    user_profiles_path = os.path.join(OUTPUT_DIR, 'user_profiles.pkl')
    risk_feature_mappings_path = os.path.join(OUTPUT_DIR, 'risk_feature_mappings.pkl')
    numerical_features_path = os.path.join(OUTPUT_DIR, 'numerical_features.pkl')
    categorical_features_for_preprocessing_path = os.path.join(OUTPUT_DIR, 'categorical_features_for_preprocessing.pkl')
    initial_df_path = os.path.join(OUTPUT_DIR, 'initial_df.pkl')

    # Tüm gerekli dosyaların var olduğundan emin ol
    required_files = {
        'Model': model_path,
        'Preprocessor': preprocessor_path,
        'Target Scaler': target_scaler_path,
        'User Profiles': user_profiles_path,
        'Risk Feature Mappings': risk_feature_mappings_path,
        'Numerical Features List': numerical_features_path,
        'Categorical Features List': categorical_features_for_preprocessing_path,
        'Initial DataFrame': initial_df_path
    }

    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"HATA: '{name}' dosyası bulunamadı: {path}. "
                "Lütfen Docker ile model eğitimini (docker run python main.py) tamamladığınızdan emin olun."
            )

    # Dosyaları yükle
    try:
        model = tf.keras.models.load_model(model_path)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
        with open(user_profiles_path, 'rb') as f:
            user_profiles = pickle.load(f)
        with open(risk_feature_mappings_path, 'rb') as f:
            risk_feature_mappings = pickle.load(f)
        with open(numerical_features_path, 'rb') as f:
            numerical_features = pickle.load(f)
        with open(categorical_features_for_preprocessing_path, 'rb') as f:
            categorical_features_for_preprocessing = pickle.load(f)
        with open(initial_df_path, 'rb') as f:
            initial_df = pickle.load(f)

        print("Tüm varlıklar başarıyla yüklendi.")

    except Exception as e:
        print(f"Varlık yükleme sırasında beklenmeyen hata: {e}")
        raise # Yükleme hatasında uygulamayı durdur

# Uygulama başladığında varlıkları yükle
with app.app_context():
    load_all_assets()

# --- Web Arayüzü (Routes) ---

@app.route('/')
def index():
    """Ana sayfayı (HTML formu) sunar."""
    # Front-end'deki select input'ları için seçenekleri HTML şablonuna gönderir
    return render_template('index.html', 
                           mfa_methods=MFA_METHODS, 
                           applications=APPLICATIONS, 
                           browsers=BROWSERS, 
                           oss=OSS, 
                           units=UNITS, 
                           titles=TITLES)

@app.route('/predict', methods=['POST'])
def predict():
    """Gelen JSON verisini işler ve risk skoru tahmini döndürür."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    
    try:
        required_keys = ['UserId', 'ClientIP', 'MFAMethod', 'Application', 'Browser', 'OS', 'Unit', 'Title']
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Eksik veri: {key}"}), 400

        # Kullanıcının profilinin varlığını kontrol et
        if data['UserId'] not in user_profiles:
            return jsonify({"error": f"Kullanıcı ID '{data['UserId']}' için profil bulunamadı. Lütfen kayıtlı bir kullanıcı ID girin."}), 400

        # Yeni giriş DataFrame'ini oluştur
        entry_df = pd.DataFrame([{
            'UserId': data['UserId'],
            'CreatedAt': datetime.now(), 
            'ClientIP': data['ClientIP'],
            'MFAMethod': data['MFAMethod'],
            'Application': data['Application'],
            'Browser': data['Browser'],
            'OS': data['OS'],
            'Unit': data['Unit'],
            'Title': data['Title'],
            'IsRisky_Scenario_Gen': 0 # Bu alan model eğitimi dışı, sabit bırakılabilir
        }])
        
        # Zaman ve IP blok özelliklerini ekle
        entry_df['CreatedAt_Hour'] = entry_df['CreatedAt'].dt.hour
        entry_df['CreatedAt_DayOfWeek'] = entry_df['CreatedAt'].dt.dayofweek
        entry_df['CreatedAt_Month'] = entry_df['CreatedAt'].dt.month
        entry_df['ClientIP_Block'] = entry_df['ClientIP'].apply(lambda x: '.'.join(x.split('.')[:-1]) + '.')

        # Risk özelliklerini hesapla ve ekle (feature_engineering.py'deki mantık)
        temp_risk_feature_mappings_api = {
            'is_ip_changed_feature': ('ip_change', 'ClientIP', 'base_ip'), 
            'is_time_anomaly_feature': ('time_anomaly', 'CreatedAt', 'avg_entry_hour'),
            'is_mfa_changed_feature': ('mfa_change', 'MFAMethod', 'preferred_mfa'),
            'is_browser_os_changed_feature': ('browser_os_change', ('Browser', 'OS'), ('preferred_browser', 'preferred_os')),
            'is_application_changed_feature': ('application_change', 'Application', 'preferred_app'),
            'is_unit_changed_feature': ('unit_change', 'Unit', 'unit'),
            'is_title_mismatch_feature': ('title_mismatch', 'Title', 'title')
        }
        
        for col_name, (feature_type, current_col, profile_key) in temp_risk_feature_mappings_api.items():
            if feature_type in ['ip_change', 'time_anomaly']:
                entry_df[col_name] = entry_df.apply(lambda row: get_risk_feature(row, user_profiles, feature_type), axis=1)
            elif isinstance(current_col, tuple):
                entry_df[col_name] = entry_df.apply(lambda row: get_risk_feature(row, user_profiles, feature_type, 
                                                                    current_value=(row[current_col[0]], row[current_col[1]]),
                                                                    profile_key=(profile_key[0], profile_key[1])), axis=1)
            else:
                entry_df[col_name] = entry_df.apply(lambda row: get_risk_feature(row, user_profiles, feature_type, 
                                                                    current_value=row[current_col], 
                                                                    profile_key=profile_key), axis=1)

        # Kural tabanlı gerçek risk skoru hesapla
        entry_df['RiskScore'] = entry_df.apply(
            lambda row: calculate_risk_score(row, RISK_WEIGHTS, temp_risk_feature_mappings_api), axis=1
        )

        # Tahmin için geçmiş verileri ve yeni girişi birleştir
        selected_user_id = data['UserId']
        user_recent_entries_df = initial_df[initial_df['UserId'] == selected_user_id].sort_values(by='CreatedAt').tail(SEQUENCE_LENGTH - 1).copy()
        
        if len(user_recent_entries_df) < SEQUENCE_LENGTH - 1:
               print(f"Uyarı: Kullanıcı {selected_user_id} için yeterli geçmiş kayıt bulunamadı. Dizinin başı sıfırlarla doldurulacaktır.")
        
        combined_df_for_prediction = pd.concat([user_recent_entries_df, entry_df], ignore_index=True)
        combined_df_for_prediction = combined_df_for_prediction.tail(SEQUENCE_LENGTH)

        # Model tahmini için veriyi hazırla ve tahmin yap
        features_to_transform = numerical_features + categorical_features_for_preprocessing
        df_for_transform = combined_df_for_prediction[features_to_transform]
        processed_data = preprocessor.transform(df_for_transform).toarray()
        
        feature_dimension = processed_data.shape[1]
        single_sequence = np.zeros((SEQUENCE_LENGTH, feature_dimension))
        actual_sequence_length = len(processed_data)
        single_sequence[SEQUENCE_LENGTH - actual_sequence_length:] = processed_data
        single_sequence_reshaped = single_sequence.reshape(1, SEQUENCE_LENGTH, feature_dimension) 
        
        predicted_scaled_score = model.predict(single_sequence_reshaped)[0][0] 
        predicted_original_score = target_scaler.inverse_transform([[predicted_scaled_score]])[0][0]

        actual_risk_score = entry_df.iloc[-1]['RiskScore'] 

        # Sonuçları yüzde olarak döndür
        predicted_original_score_percent = predicted_original_score * 100
        actual_risk_score_percent = actual_risk_score * 100

        risk_evaluation = "Yüksek Riskli" if predicted_original_score > 0.50 else "Düşük Riskli"

        return jsonify({
            "userId": data['UserId'],
            "actualRiskScore": round(actual_risk_score_percent, 2),
            "predictedRiskScore": round(predicted_original_score_percent, 2),
            "isRisky": risk_evaluation
        })

    except Exception as e:
        app.logger.error(f"Tahmin sırasında hata oluştu: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Flask uygulamasını başlat. host='0.0.0.0' Docker içinde önemlidir.
    app.run(debug=True, host='0.0.0.0', port=5000)