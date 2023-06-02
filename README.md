# ✈ Konlpy와 TF-IDF를 이용한 한국어 문장 유사도 계산 API (Flask) 
Konlpy와 TF-IDF를 이용한 한국어 문장 유사도 계산 API & CLOVA OCR API 서버

## ✈ Run Project
### .env setting
```
FLASK_API_KEY={OPENAI secret API key}
AWS_ACCESS_KEY={AWS access key}
AWS_SECRET_KEY={AWS secret key}
X_OCR_SECRET={CLOVA OCR secret key}
REDIS_PW={redis password}
```
### Run Server using docker-compose
```
sudo docker-compose -f /{설치경로}/docker-compose.yml up --build -d 
```

### Precondition (pip install --no-cache-dir -r requirements.txt로 대체 가능)

```
python : 3.8
JAVA : jdk-19
JAVA_HOME 환경 변수 설정 필수 (ex. C:\Program Files\Java\jdk-19 )
JPype : 1.1.2버전 (JPype1-1.1.2-cp38-cp38-win_amd64.whl)
```


## ✈ Stacks


 
### Konlpy
![konlpy](https://user-images.githubusercontent.com/81161750/206637077-f06d2eb3-2fc5-45b8-af7f-c9ca462cbed2.png)

* reference : [파이썬 한국어 NLP — KoNLPy 0.6.0 documentation](https://konlpy.org/ko/latest/index.html)
### CLOVA OCR
![image](https://github.com/MIDUBANG/Korean-Sentence-Similarity-API/assets/87990290/0c48f0da-8b1a-4777-a7a5-cd9319c15f4d)
<br>
<br>

### ETC
<img src="https://img.shields.io/badge/Flask-3481FE?style=for-the-badge&logo=Flask&logoColor=white">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
<img src="https://img.shields.io/badge/Amazon AWS-FF9900?style=for-the-badge&logo=AmazonAWS&logoColor=white">


<br>
