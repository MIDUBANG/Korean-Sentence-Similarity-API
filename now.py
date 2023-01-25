import os
from flask import Flask, render_template
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

vectorizer = TfidfVectorizer(min_df=1, decode_error="ignore")


@app.route("/api/nlp", methods=["POST"])
def nlp():
    input = request.get_json()

    # print(input)
    # contents = input['contents']

    # ✅ 기존 특약 데이터 배열
    
    contents = [
        ["반려동물을 키우지 않는다. 금지", "강아지를 키우면 벽지 도배비를 지불한다.", "고양이를 키우지 않는다."],
        [" 아 하기 싫다 아 하기 싫다", "밥 먹고싶다 아 하기 싫다", "졸라실다 아 하기 싫다"],
        ["지겹다아 하기 싫다", "아무말 대잔치 아 하기 싫다", "뭐쓰지 아 하기 싫다"],
    ]


    contents_tokens = []
    temp = []
    for case in contents:
        for row in case:
            temp.append(t.morphs(row))
        contents_tokens.append(temp)
        temp = []

    contents_for_vectorize = []
    temp = []

    for case in contents_tokens:  # 토큰화 다시 붙임
        for case_ in case:
            sentence = ""
            for token in case_:
                sentence += " " + token
            temp.append(sentence)
        contents_for_vectorize.append(temp)
        temp = []

    # 다시 붙인거 contents_for_vectorize
    #[[' 반려동물 을 키우지 않는다 . 금지', ' 강아지 를 키우면 벽지 도배 비 를 지불 한다 .', ' 고양이 를 키우지 않는다 .'], [' 아 하기 싫다 아 하기 싫다', ' 밥 먹고싶다 아 하기 싫다', ' 졸라실다 아 하기 싫다'], [' 지겹다아 하기 싫다', ' 아무 말 대 잔치 아 하기 싫다', ' 뭐 쓰지 아 하기 싫다']]

    X = []

    for i in range(3):
        X.append(vectorizer.fit_transform(contents_for_vectorize[i]))
    #vectorizer.get_feature_names()
    # 0 ['강아지', '고양이', '금지', '도배', '반려동물', '벽지', '않는다', '지불', '키우면', '키우지', '한다']
    # 1 ['먹고싶다', '싫다', '졸라실다', '하기']
    # 2 ['싫다', '쓰지', '아무', '잔치', '지겹다아', '하기']

    num_samples = [] #(문장 개수, 토큰 개수)
    for i in range(3):
        num_samples.append(X[i].shape)



    # ✅ 입력으로 들어온 특약 배열
    new_post = ["임대차 계약해지", "임대어쩌구", "반려동물키우기금지"]

    new_post_tokens = [t.morphs(row) for row in new_post]

    new_post_for_vectorize = []

    for content in new_post_tokens:
        sentence = ""
        for word in content:
            sentence = sentence + " " + word
        new_post_for_vectorize.append(sentence)
    # new_post_for_vectorize :[' 임대차 계약해지', ' 임대 어 쩌구', ' 반려동물 키우기 금지']
   
    new_post_vec = vectorizer.transform(new_post_for_vectorize)  # 출력해도 안보임

    #  1️⃣ 거리 합의 최소 계산

    d = 0
    min = 100
    min_index = 0
    res = []
    d_sum = []
    

    post_vec = X[0].getrow(0) 
    d = dist_raw(post_vec, new_post_vec[0])
    print("d", d)
    

    """

      for i in range(3):  # 3
        post_vec = X[1].getrow(1) 
        d = dist_raw(post_vec, new_post_vec[2])
        print("d", d)


    for a in range(len(new_post)):# 새로 들어온거
        for i in range(len(X)): # X의 개수
            for j in range(num_samples[1][0]): # x의 num_samples 개수 
                    post_vec = X[1].getrow(j) # 기존꺼 

                    d = dist_raw(post_vec, new_post_vec[a]) 

                    print('test',i,j,'post_vec',post_vec)

                    if min>d:                
                        min = d
                        min_index = i

        # print(j, "번 문장과 가장 가까운 case는 ", min_index, "거리는 ", d)
    """

    return jsonify({"w": "1"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
