# config.py

import os
from faker import Faker

# Temel Yapılandırma
NUM_USERS = 100
ENTRIES_PER_USER = 50
RISK_INJECTION_RATE = 0.15 # Riskli senaryoların oranı (örneğin %15)
SEQUENCE_LENGTH = 5 # LSTM için zaman serisi uzunluğu

# Sabitler (Veri Çeşitliliği İçin)
MFA_METHODS = ['SMS_OTP', 'Email_OTP', 'App_Auth', 'Hardware_Token']
APPLICATIONS = ['AppA', 'AppB', 'AppC', 'AppD']
BROWSERS = ['Chrome', 'Firefox', 'Safari', 'Edge']
OSS = ['Windows', 'macOS', 'Linux', 'iOS', 'Android']
UNITS = ['HR', 'Finance', 'Engineering', 'Marketing', 'Sales']
TITLES = ['Manager', 'Analyst', 'Director', 'Specialist', 'Associate']

# Risk Ağırlıkları (Kural Tabanlı Sistem İçin)
RISK_WEIGHTS = {
    'ip_change': 0.35,
    'time_anomaly': 0.25,
    'mfa_change': 0.15,
    'browser_os_change': 0.10,
    'application_change': 0.05,
    'unit_change': 0.05,
    'title_mismatch': 0.05
}

# Faker objesi (sahte veri üretimi için)
fake = Faker()

# Çıktı klasörü (eğer kaydedilecek dosyalar varsa)
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)