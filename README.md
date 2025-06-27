# Risk Tahmin Sistemi (Risk Prediction System)  
**Son Güncelleme**: 09.06.2025  

---

## 🌟 Proje Amacı / Project Purpose  
**TR**: Bu proje, kullanıcıların oturum açma davranışlarını analiz ederek potansiyel güvenlik risklerini tahmin etmek için makine öğrenimi tabanlı bir sistem geliştirir. LSTM modeli ile anormallikleri tespit eder ve Flask + Docker ile dağıtılır.  

**EN**: This project develops an ML-based system to predict security risks by analyzing user login behavior. It detects anomalies using an LSTM model and is deployed via Flask + Docker.  

---

## 🛠️ Teknolojiler / Technologies  
- **Python 3.11**  
- **TensorFlow/Keras** (LSTM Model)  
- **Flask** (Web API)  
- **Docker** (Containerization)  
- **Scikit-learn** (Feature Engineering)  
- **Pandas/Numpy** (Data Processing)  

---

## 📊 Veri Akışı / Data Pipeline  
1. **Mock Veri Üretimi** → 2. **Özellik Mühendisliği** → 3. **Sıralı Ön İşleme** → 4. **LSTM Model Eğitimi** → 5. **Risk Tahmini API**  

---

## 📌 Temel Özellikler / Key Features  
### 🔍 Kural Tabanlı Risk Skoru / Rule-Based Risk Scoring  
**TR**:  
- IP, MFA, zaman sapması gibi 7+ risk faktörüne dayalı ağırlıklı skorlama  
- Örnek: `is_ip_changed_feature`, `is_time_anomaly_feature`  

**EN**:  
- Weighted scoring based on 7+ risk factors (IP, MFA, time deviation)  
- Example: `is_ip_changed_feature`, `is_time_anomaly_feature`  

### 🤖 Model Performansı / Model Performance  
```python
Test R2 Score: 0.9932  # %99.32 varyans açıklaması
MAE: 0.008             # Düşük tahmin hatası
TR: Model, sentetik veride davranışsal kalıpları yakalamada mükemmel performans gösterdi.
EN: The model excelled at capturing behavioral patterns in synthetic data.
```
🚀 Kurulum / Installation
Docker ile Çalıştırma / Run with Docker
bash
docker build -t risk-prediction .  
docker run -p 5000:5000 risk-prediction
TR: Uygulama http://localhost:5000 adresinde başlar.
EN: App runs at http://localhost:5000.

🖥️ API Endpoint
POST /predict
```
json
{
  "UserId": "U1001",
  "ClientIP": "192.168.1.100",
  "MFAMethod": "SMS"
}
```
Yanıt / Response:
```
json
{
  "predictedRisk": "Yüksek (92%)",
  "ruleBasedRisk": "85%"
}
```
📂 Proje Yapısı / Project Structure
text
├── /output/            # Eğitilmiş model ve ön işlemciler
├── data_generator.py    # Mock veri üreteci
├── feature_engineering.py # Risk özellikleri mühendisliği
├── model_builder.py     # LSTM modeli oluşturma
├── app.py              # Flask API
└── Dockerfile          # Çok aşamalı container build
🌍 Kullanım Senaryoları / Use Cases
TR: Şüpheli oturum açma girişimlerinin gerçek zamanlı tespiti

EN: Real-time detection of suspicious login attempts

TR: Kullanıcı davranışı anomalisi uyarıları

EN: User behavior anomaly alerts

📜 Lisans / License
MIT - Proje özgürce kullanılabilir ve modifiye edilebilir.
MIT - Free to use and modify.

👨💻 Geliştirici / Developer: Serkan Öğütcüoğlu
