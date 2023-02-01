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

import numpy as np

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


initialData = [
        ["반려동물을 키우지 않는다. 금지", "강아지를 키우면 벽지 도배비를 지불한다.", "고양이를 키우지 않는다."],
        ["입주 후 생긴 하자는 세입자가 수리한다.", "하자는 임차인이 수리한다.", "하자는 임차인이 책임지고 수리"],
        ["월세를 밀리면 퇴거", "이유 없이 월세를 두 달 이상 밀리면 퇴실", "월세를 내지 못한 경우 퇴실한다."],
    ]


distance_array = []

def get_best(case_num,input):
    contents = initialData[case_num] # ✅ 기존 특약 데이터 배열

    contents_tokens = [t.morphs(row) for row in contents]

    contents_for_vectorize = []
    for content in contents_tokens:
        sentence = ''
        for word in content:
            sentence = sentence + ' ' + word
        contents_for_vectorize.append(sentence)

    #print(contents_for_vectorize) 
    # [' 임대차 계약 해지', ' 임대 어쩌구', ' 반려동물 키우기 금지']
   
    X = vectorizer.fit_transform(contents_for_vectorize)
    #print('1', vectorizer.get_feature_names()) 
    # ['계약', '금지', '반려동물', '어쩌구', '임대', '임대차', '키우기', '해지']
    

    num_samples, num_features = X.shape 
    # X
    #(0, 0) 1.0
    #(1, 0) 1.0
    #(2, 0) 1.0

    #new_post = [input['text']]
    new_post = input# ✅ 입력으로 들어온 특약 배열
    new_post_tokens = [t.morphs(row) for row in new_post] # [['입력', '된', '문', '장임'], ['두번째', '문장'], ['세번', '째', '분장']]

    new_post_for_vectorize = []

    for content in new_post_tokens:
        sentence = ''
        for token in content:
            sentence +=' ' + token
        new_post_for_vectorize.append(sentence)  #  [' 입력 된 문 장임', ' 입력 된 문 장임', ' 입력 된 문 장임']

    #print(new_post_for_vectorize)

    new_post_vec = vectorizer.transform(new_post_for_vectorize)  # 출력해도 안보임


    best_dist = 65535
    best_i = None
    
    res = []

    # ✅ 거리 계산

    d = 0
    min = 100
    min_index = 0
    res = []
    
    temp_distance = []
    for i in range(0, num_samples):
        post_vec = X.getrow(i) 

        d = dist_raw(post_vec, new_post_vec)
        temp_distance.append(d)

        #print(case_num,'case와 인풋',i,'사이 거리는',d)
        
    data = np.array(temp_distance)
    min_val = np.min(data)
    print(case_num,'번 케이스와의 최소 거리',min_val)
    return min_val





@app.route('/api/nlp', methods=['POST'])
def nlp():
    input = (request.get_json())
    contents = input['contents']
    extraInfo = input['extraInfo']

    print(contents)
    print(extraInfo)

    #contents = ['반려동물을 키우지 않는다. 금지','월세 밀리면 퇴거합니다.','입주 후 생긴 하자는 임차인이 수리한다.']
    

    # ✅ in
    min_dis = 1000
    best_case_i = 999
    answer_in = []
    for i in range(3): # input 3개, 세번 돌린다. 
        for j in range(len(initialData)): # 보유 중인 case (3개 라고 가정)
            sum_of_distance = get_best(j,[contents[i]]) # 인풋 중 하나만
            if min_dis > sum_of_distance:
                min_dis = sum_of_distance
                best_case_i = j
        print(best_case_i)
        min_dis = 1000
        if min_dis < 0.7: # 거리 기준
            answer_in.append(best_case_i)

    # ✅ out
    answer_out = []

    # 1) 공통 필수인데 안들어간 것
    essential = [34,35,36,37,38]
    for es in essential:
        if not es in answer_in:
            answer_out.append(es)
  

    # 2) 선택적
    pet = extraInfo['pet']
    loan = extraInfo['loan']
    substitute = extraInfo['substitute']

    if pet: # 반려 동물
        if not 77 in answer_in:
            answer_out.append(77)
    if loan: # 전세 대출
        if not 88 in answer_in:
            answer_out.append(88)
    if substitute: # 대리인
        if not 99 in answer_in:
            answer_out.append(99)
    

    # ✅ 복비 계산
    monthly = extraInfo['monthly'] #월세or전세
    commission = extraInfo['commission'] #복비
    deposit = extraInfo['deposit'] #보증금 (전세금)
    monthlyMoney = extraInfo['monthlyMoney'] #월세

    # 1) 거래금액 Scale 계산하기 
    if monthly: #월세
        if deposit + monthlyMoney*100 <= 50000000: #5천만원 이하면
            scale = deposit + monthlyMoney*70
        else:
            scale = deposit + monthlyMoney*100
    else: #전세
        if deposit < 50000000:
            scale= deposit

    # 2) 거래금액에 따른 상한요율 rate, 한도액 limit 계산
    if  scale < 50000000: #5천만원 미만 / 0.5 (rate) / 20만(limit)
        rate = 0.005
        limit = 200000
    elif scale < 100000000: #5천 이상, 1억 미만 / 0.4 / 30만
        rate = 0.004
        limit = 300000
    elif scale < 600000000: # 1억 이상, 6억 미만 / 0.3 / 없음
        rate = 0.003
        limit = float('inf')
    elif scale < 1200000000: # 6억 이상, 12억 미만 / 0.4 / 없음
        rate = 0.004
        limit = float('inf')
    elif scale < 1500000000: # # 1억 이상, 6억 미만 / 0.5 / 없음
        rate = 0.005
        limit = float('inf')
    else: # 1억 이상, 6억 미만 / 0.6 / 없음
        rate = 0.006
        limit = float('inf')

    # 3) 최대 복비 계산
    answer_commission = scale * rate
    if answer_commission > limit:
        answer_commission = limit

    # 4) 바가지 당첨
    is_expensive = False
    if answer_commission < commission:
        is_expensive = True

    return jsonify({"in": answer_in,"out":answer_out, "answer_commission":answer_commission,"is_expensive":is_expensive})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)