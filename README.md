# ✈ Konlpy와 TF-IDF를 이용한 한국어 문장 유사도 계산 API (Flask) 
Konlpy와 TF-IDF를 이용한 한국어 문장 유사도 계산 API 서버

## URL 
https://midubang.com/



## ✈ Run Project
### Run Server
```
git clone https://github.com/MIDUBANG/Korean-Sentence-Similarity-API.git
pip install -r requirements.txt
flask run 
```

### Precondition

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

<br>
<br>

### ETC
<img src="https://img.shields.io/badge/Flask-3481FE?style=for-the-badge&logo=Flask&logoColor=white">
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
<img src="https://img.shields.io/badge/Amazon AWS-FF9900?style=for-the-badge&logo=AmazonAWS&logoColor=white">


<br>

### ✈ Modul Script 

```
def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


t = Okt()

vectorizer = TfidfVectorizer(min_df=1, decode_error='ignore')


@app.route('/api/nlp', methods=['POST'])
def nlp():
    input = (request.get_json())
    print(input)
    contents = input['contents']
    contents_tokens = [t.morphs(row) for row in contents]
    contents_for_vectorize = []
    for content in contents_tokens:
        sentence = ''
        for word in content:
            sentence = sentence + ' ' + word
        contents_for_vectorize.append(sentence)

    X = vectorizer.fit_transform(contents_for_vectorize)
    num_samples, num_features = X.shape

    new_post = [input['text']]
    new_post_tokens = [t.morphs(row) for row in new_post]
    new_post_for_vectorize = []
    for content in new_post_tokens:
        sentence = ''
        for word in content:
            sentence = sentence + ' ' + word
        new_post_for_vectorize.append(sentence)
    new_post_vec = vectorizer.transform(new_post_for_vectorize)
    
    best_dist = 65535
    best_i = None

    res = []
    for i in range(0, num_samples):
        post_vec = X.getrow(i)
        d = dist_raw(post_vec, new_post_vec)

        res.append({'i' : i, 'distance' : d, 'content': contents[i]})
        print(d, best_dist)
        if d < best_dist:
            best_dist = d
            best_i = i

    res.append({'best_i':best_i,'best_distance' :best_dist, 'content':contents[best_i], 'target':new_post})
    return jsonify(res)```
