# Dockerfile

# Aşama 1: Model Eğitimi ve Varlıkların Oluşturulması
FROM python:3.11-slim-buster as builder

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli tüm Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını (model eğitimi için gerekli olanlar dahil) kopyala
COPY . .

# Modellerin ve diğer varlıkların kaydedileceği klasörü oluştur
RUN mkdir -p /app/output

# Model eğitimini çalıştır. Bu adım, modelleri /app/output içine kaydeder.
# Bu aşama, Docker build sırasında gerçekleşir.
RUN python main.py

# Aşama 2: Web Uygulamasının Oluşturulması (Daha Küçük, Optimize Edilmiş İmaj) 
FROM python:3.11-slim-buster

# Çalışma dizinini ayarla
WORKDIR /app

# Sadece Flask uygulamasının çalışması için gereken bağımlılıkları yükle.
# Eğer tüm bağımlılıklar aynıysa, builder aşamasından tekrar yüklemek yerine
# kopyalayabiliriz, ama bu şekilde daha güvenlidir.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flask uygulamasını ve diğer gerekli modülleri kopyala
# (main.py, data_generator.py, model_builder.py, vb. artık app.py için gerekmeyenler kopyalanmaz)
COPY app.py .
COPY config.py .
COPY feature_engineering.py .
COPY preprocessing.py .
COPY templates/ templates/

# Eğitilmiş model ve varlıkları "builder" aşamasından kopyala
COPY --from=builder /app/output /app/output

# Uygulamanın çalışacağı portu belirt
EXPOSE 5000

# Flask uygulamasını başlatma komutu
CMD ["python", "app.py"]