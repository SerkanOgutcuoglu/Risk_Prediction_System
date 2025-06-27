# feature_engineering.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# get_risk_feature fonksiyonu
def get_risk_feature(entry_row, user_profiles_dict, feature_type, current_value=None, profile_key=None):

    user_profile = user_profiles_dict.get(entry_row['UserId'])
    if not user_profile:
        # Kullanıcı profili yoksa veya UserId bulunamazsa risk yok varsay
        # Bu durum genelde test amaçlı tekil girişlerde olabilir eğer profil oluşturulmamışsa
        return 0 

    is_risky = 0

    if feature_type == 'ip_change':
        # IP blok değişikliği için
        entry_ip_block = '.'.join(entry_row['ClientIP'].split('.')[:-1]) + '.'
        # user_profile['base_ip'] string formatında, kontrol et
        profile_ip_block = '.'.join(user_profile['base_ip'].split('.')[:-1]) + '.'
        if entry_ip_block != profile_ip_block:
            is_risky = 1
    
    elif feature_type == 'time_anomaly':
        # Zaman anormalliği (gece saati veya hafta sonu)
        entry_hour = entry_row['CreatedAt'].hour
        entry_day_of_week = entry_row['CreatedAt'].weekday() # Pazartesi=0, Pazar=6

        # Gece saati (00:00-06:00 arası veya 22:00-24:00 arası)
        is_night_time = (entry_hour >= 0 and entry_hour < 6) or (entry_hour >= 22 and entry_hour <= 23)
        # Hafta sonu (Cumartesi=5, Pazar=6)
        is_weekend = (entry_day_of_week == 5) or (entry_day_of_week == 6)

        if is_night_time or is_weekend:
            is_risky = 1

    elif feature_type == 'mfa_change':
        # MFA yöntemi değişikliği
        if entry_row['MFAMethod'] != user_profile['preferred_mfa']:
            is_risky = 1

    elif feature_type == 'browser_os_change':
        # Tarayıcı veya İşletim Sistemi değişikliği

        if isinstance(current_value, tuple) and isinstance(profile_key, tuple):
            entry_browser, entry_os = current_value
            profile_browser, profile_os = user_profile[profile_key[0]], user_profile[profile_key[1]]
            if (entry_browser != profile_browser) or (entry_os != profile_os):
                is_risky = 1
        elif (entry_row['Browser'] != user_profile['preferred_browser']) or \
             (entry_row['OS'] != user_profile['preferred_os']): # Fallback eğer current_value geçirilmezse
            is_risky = 1

    elif feature_type == 'application_change':
        # Uygulama değişikliği
        if entry_row['Application'] != user_profile['preferred_app']:
            is_risky = 1

    elif feature_type == 'unit_change':
        # Birim değişikliği
        if entry_row['Unit'] != user_profile['unit']:
            is_risky = 1

    elif feature_type == 'title_mismatch':
        # Unvan uyuşmazlığı
        if entry_row['Title'] != user_profile['title']:
            is_risky = 1
            
    return is_risky


def apply_feature_engineering(df, user_profiles):

    print("\n--- Özellik Mühendisliği ve Kural Tabanlı Risk Etiketleme Başlıyor ---")
    
    # Risk özelliklerini ve bunların nasıl haritalandırılacağını tanımla
    risk_feature_mappings = {
        'is_ip_changed_feature': ('ip_change', 'ClientIP', 'base_ip'), # ClientIP ve base_ip'i get_risk_feature kendi işler
        'is_time_anomaly_feature': ('time_anomaly', 'CreatedAt', 'avg_entry_hour'), # avg_entry_hour doğrudan kullanılmıyor, ancak tutarlılık için var
        'is_mfa_changed_feature': ('mfa_change', 'MFAMethod', 'preferred_mfa'),
        'is_browser_os_changed_feature': ('browser_os_change', ('Browser', 'OS'), ('preferred_browser', 'preferred_os')),
        'is_application_changed_feature': ('application_change', 'Application', 'preferred_app'),
        'is_unit_changed_feature': ('unit_change', 'Unit', 'unit'),
        'is_title_mismatch_feature': ('title_mismatch', 'Title', 'title')
    }

    # Her bir risk özelliğini DataFrame'e ekle
    for col_name, (feature_type, current_col, profile_key) in risk_feature_mappings.items():
        if feature_type in ['ip_change', 'time_anomaly']: 
            df[col_name] = df.apply(lambda row: get_risk_feature(row, user_profiles, feature_type), axis=1)
        elif isinstance(current_col, tuple): # Birden fazla sütuna bağlı olanlar
            df[col_name] = df.apply(lambda row: get_risk_feature(row, user_profiles, feature_type, 
                                                                 current_value=(row[current_col[0]], row[current_col[1]]),
                                                                 profile_key=(profile_key[0], profile_key[1])), axis=1)
        else: # Tek bir sütuna bağlı olanlar
            df[col_name] = df.apply(lambda row: get_risk_feature(row, user_profiles, feature_type, 
                                                                 current_value=row[current_col], 
                                                                 profile_key=profile_key), axis=1)
    


    return df, risk_feature_mappings


def calculate_risk_score(row, weights, risk_feature_mappings):

    score = 0.0

    # Risk özelliklerini içeren sütunlar ve weights'deki karşılıkları için bir mapping
    feature_to_weight_key = {
        'is_ip_changed_feature': 'ip_change',
        'is_time_anomaly_feature': 'time_anomaly',
        'is_mfa_changed_feature': 'mfa_change',
        'is_browser_os_changed_feature': 'browser_os_change',
        'is_application_changed_feature': 'application_change',
        'is_unit_changed_feature': 'unit_change',
        'is_title_mismatch_feature': 'title_mismatch'
    }

    for feature_col_name, weight_key in feature_to_weight_key.items():
        # Eğer row'da bu özellik sütunu varsa VE değeri 1 ise (yani riskli ise)
        if feature_col_name in row and row[feature_col_name] == 1:
            if weight_key in weights: # İlgili ağırlık config'de tanımlı mı kontrol et
                score += weights[weight_key]

    
    return score