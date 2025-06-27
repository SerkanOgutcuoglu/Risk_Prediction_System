docker pull ---> repodaki image çekmek için.

docker pull secode/risk-prediction-app:latest

docker run ---> başlatmak için

docker run -d -p 5000:5000 --name risk-prediction-web-container secode/risk-prediction-app:latest

web içinde;

http://127.0.0.1:5000


web üzerinden kapatsanız bile container çalışmaya devam edecektir.


durdurma

docker stop risk-prediction-web-container

container silme

docker rm risk-prediction-web-container

image silme

docker rmi secode/risk-prediction-app:latest