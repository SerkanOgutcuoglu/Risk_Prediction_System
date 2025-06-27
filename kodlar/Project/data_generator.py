# data_generator.py
import os
import pandas as pd
import random
from datetime import datetime, timedelta
import pickle # user_profiles'ı kaydetmek için

from config import NUM_USERS, ENTRIES_PER_USER, RISK_INJECTION_RATE, \
                   MFA_METHODS, APPLICATIONS, BROWSERS, OSS, UNITS, TITLES, fake, OUTPUT_DIR

def generate_mock_data():
    """
    Sahte kullanıcı profilleri ve giriş kayıtları oluşturur.
    """
    user_profiles = {}
    all_entries = []

    print("Kullanıcı profilleri ve giriş kayıtları oluşturuluyor...")

    for i in range(NUM_USERS):
        user_id = f'U{10000 + i}'
        
        # Kullanıcı Profili Oluşturma
        base_ip = '.'.join(fake.ipv4_public().split('.')[:-1]) + '.'
        preferred_mfa = random.choice(MFA_METHODS)
        preferred_app = random.choice(APPLICATIONS)
        preferred_browser = random.choice(BROWSERS)
        preferred_os = random.choice(OSS)
        unit = random.choice(UNITS)
        title = random.choice(TITLES)
        
        # Ortalama giriş saati (09:00 - 17:00 arası)
        avg_entry_hour = random.randint(9, 17) 

        user_profiles[user_id] = {
            'base_ip': base_ip,
            'preferred_mfa': preferred_mfa,
            'preferred_app': preferred_app,
            'preferred_browser': preferred_browser,
            'preferred_os': preferred_os,
            'unit': unit,
            'title': title,
            'avg_entry_hour': avg_entry_hour
        }

        # Giriş Kayıtları Oluşturma
        for j in range(ENTRIES_PER_USER):
            created_at = datetime.now() - timedelta(days=random.randint(0, 365), hours=random.randint(0, 23), minutes=random.randint(0, 59))
            
            is_risky_scenario = random.random() < RISK_INJECTION_RATE

            entry_ip = fake.ipv4_public()
            entry_mfa = random.choice(MFA_METHODS)
            entry_app = random.choice(APPLICATIONS)
            entry_browser = random.choice(BROWSERS)
            entry_os = random.choice(OSS)
            entry_unit = random.choice(UNITS)
            entry_title = random.choice(TITLES)


            if is_risky_scenario:
                risk_type = random.choice(['ip', 'time', 'mfa', 'browser_os', 'app', 'unit', 'title'])
                if risk_type == 'ip':
                    entry_ip = fake.ipv4_public() # Farklı IP
                elif risk_type == 'time':
                    # Anormal saatler: gece (00-06) veya çok geç (22-24)
                    # VEYA hafta sonu
                    
                    if random.random() < 0.5: # Anormal saat
                        entry_hour = random.choice(list(range(0, 6)) + list(range(22, 24)))
                        created_at = created_at.replace(hour=entry_hour) # Sadece saati değiştir
                    else: # Hafta sonu
                        # Mevcut günün haftanın hangi günü olduğunu bul
                        current_weekday = created_at.weekday() # Pazartesi=0, Pazar=6
                        
                        # Eğer zaten hafta sonu değilse, en yakın hafta sonuna git
                        if current_weekday < 5: # Pazartesi-Cuma ise
                            days_to_saturday = 5 - current_weekday
                            created_at = created_at + timedelta(days=days_to_saturday)
                        # Eğer zaten hafta sonu ise (Cumartesi veya Pazar), bir şey yapmaya gerek yok
                        # veya ek olarak bir sonraki hafta sonuna atlanabilir
                        
                        # Hafta sonu bir saat de rastgele ayarlanabilir
                        created_at = created_at.replace(hour=random.randint(0, 23))


                elif risk_type == 'mfa':
                    entry_mfa = random.choice([m for m in MFA_METHODS if m != preferred_mfa] or [preferred_mfa])
                elif risk_type == 'browser_os':
                    entry_browser = random.choice([b for b in BROWSERS if b != preferred_browser] or [preferred_browser])
                    entry_os = random.choice([o for o in OSS if o != preferred_os] or [preferred_os])
                elif risk_type == 'app':
                    entry_app = random.choice([a for a in APPLICATIONS if a != preferred_app] or [preferred_app])
                elif risk_type == 'unit':
                    entry_unit = random.choice([u for u in UNITS if u != unit] or [unit])
                elif risk_type == 'title':
                    entry_title = random.choice([t for t in TITLES if t != title] or [title])
            else:
                # Normal senaryo için profil değerlerini kullan
                entry_ip = base_ip + str(random.randint(1, 254))
                entry_mfa = preferred_mfa
                entry_app = preferred_app
                entry_browser = preferred_browser
                entry_os = preferred_os
                entry_unit = unit
                entry_title = title
            
            all_entries.append({
                'UserId': user_id,
                'CreatedAt': created_at,
                'ClientIP': entry_ip,
                'MFAMethod': entry_mfa,
                'Application': entry_app,
                'Browser': entry_browser,
                'OS': entry_os,
                'Unit': entry_unit,
                'Title': entry_title,
                'IsRisky_Scenario_Gen': 1 if is_risky_scenario else 0 # Mock verideki risk etiketi
            })

    df = pd.DataFrame(all_entries)
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
    # Zaman serisi için sıralama çok önemli
    df = df.sort_values(by=['UserId', 'CreatedAt']).reset_index(drop=True)

    # user_profiles'ı kaydet
    with open(os.path.join(OUTPUT_DIR, 'user_profiles.pkl'), 'wb') as f:
        pickle.dump(user_profiles, f)

    print(f"Toplam {len(df)} giriş kaydı ve {len(user_profiles)} kullanıcı profili oluşturuldu.")
    return df, user_profiles