import os
from flask import Flask,  render_template
from flask import request, jsonify

from konlpy.tag import Okt
from flask_cors import CORS

from sklearn.feature_extraction.text import CountVectorizer
import konlpy
from konlpy.tag import Okt
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)

CORS(app)

@app.route("/") 
def home():
    return render_template("index.html")

def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


t = Okt()

vectorizer = TfidfVectorizer(min_df=1, decode_error='ignore')


@app.route('/api/nlp', methods=['POST'])
def nlp():
    input = (request.get_json())
  
    #print(input)
    #contents = input['contents']
    contents = ['임대차 계약 해지','임대 어쩌구','반려동물 키우기 금지'] # ✅ 기존 특약 데이터 배열
    contents_tokens = [t.morphs(row) for row in contents]

    contents_for_vectorize = []
    for content in contents_tokens:
        sentence = ''
        for word in content:
            sentence = sentence + ' ' + word
        contents_for_vectorize.append(sentence)

    X = vectorizer.fit_transform(contents_for_vectorize)
    print('1', vectorizer.get_feature_names())
    


    num_samples, num_features = X.shape 
    # X
    #(0, 0) 1.0
    #(1, 0) 1.0
    #(2, 0) 1.0

    #new_post = [input['text']]
    new_post = ['반려동물을 키우지 않는다. 금지','입력된 문장임','입력된 문장임','입력된 문장임'] # ✅ 입력으로 들어온 특약 배열
    new_post_tokens = [t.morphs(row) for row in new_post] # [['입력', '된', '문', '장임'], ['두번째', '문장'], ['세번', '째', '분장']]

    new_post_for_vectorize = []

    for content in new_post_tokens:
        sentence = ''
        for token in content:
            sentence +=' ' + token
        new_post_for_vectorize.append(sentence)  #  [' 입력 된 문 장임', ' 입력 된 문 장임', ' 입력 된 문 장임']


    new_post_vec = vectorizer.transform(new_post_for_vectorize)  # 출력해도 안보임


    best_dist = 65535
    best_i = None
    
    res = []

    # ✅ 거리 계산

    d = 0
    min = 100
    min_index = 0
    res = []
    
    for j in range(len(new_post)): # 새로 들어온거
        for i in range(0, num_samples):
            post_vec = X.getrow(i)

            d = dist_raw(post_vec, new_post_vec[j])
            if min > d:
                min = d
                min_index = i
        print(j,'번 문장과 가장 가까운 case는 ', min_index, '거리는 ',d )
            
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)