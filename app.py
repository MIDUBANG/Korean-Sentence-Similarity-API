import openai
import os
from flask import Flask, render_template
from flask import request, jsonify
from konlpy.tag import Okt
from flask_cors import CORS
from konlpy.tag import Okt
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle


app = Flask(__name__)
test = os.getenv('FLASK_API_KEY')

CORS(app)

YOUR_API_KEY = test
print("제발",YOUR_API_KEY)

def chatGPT(prompt, API_KEY=YOUR_API_KEY):
    # set api key
    openai.api_key = API_KEY
    # Call the chat GPT API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return completion["choices"][0]["message"]["content"].encode("utf-8").decode()


@app.route("/")
def home():
    return render_template("index.html")


def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


t = Okt()

vectorizer = TfidfVectorizer(min_df=1, decode_error="ignore")


# case 순서대로 쭉 나열
initialData = [
    [
        "월세와 보증금은 매년 시세에 맞게 조정할 수 있다.",
        "시세에 맞게 월세를 인상한다.",
        "시세에 맞게 보증금을 인상한다.",
        "금리 인상에 맞추어 월세를 인상한다.",
        "계약을 연장할 경우 월세를 인상한다."
    ],
    [
        "계약 기간을 1년으로 정했다면 1년 뒤 퇴거한다.",
        "계약 종료일 한 달 전에 해지 통보를 하지 않으면 종전과 같은 조건으로 1년 더 연장되는 것으로 본다",
        "임차인은 1년 뒤 퇴거한다. ",
        "임차인은 2년 뒤 퇴거한다.",
        "세입자는 1년 뒤 퇴거한다.",
        "세입자는 1년 뒤 무조건 집을 뺀다."
    ],
    [
        "1. 계약 만기 5개월 전 까지 재계약 의사를 밝히지 않은 경우, 계약은 만료되는 것으로 간주한다.",
        "2. 계약 연장은 계약 만기 3달 전까지 갱신 의사를 밝혀야만 가능하다.",
        "3. 묵시적 갱신은 배제한다. ",
        "4. 묵시적 갱시는 없는 것으로 한다.",
        "5. 계약 연장 없이 무조건 n년 n월 n일에 집을 뺀다."
    ],
    [
        "1. 임차인이 임차료를 연체하면 임대인이 임차인의 모든 짐을 처분한다.",
        "2. 월세를 두 번 이상 밀리면 임대인이 임차인의 짐을 강제로 뺄 수 있다.",
        "3. 집주인이 세입자의 짐을 뺼 수 있다.",
        "4. 월세를 세번 이상 밀리면 세입자 짐은 강제로 처분 될 수 있다. ",
        "5. 월세를 3회분 이상 밀리면 임대인이 임차인의 동의 없이 짐을 처분 할 수 있다."
    ],
    [
        "1. 임대인이 매매로 변경되면 임대차계약도 같이 해지된다.",
        "2.  임대인이 바뀌었을 경우, 양도인인 임대인과 임차인 사이 당연승계를 배제한다.",
        "3. 임대인이 바뀌는 경우, 이 계약은 해지된다. ",
        "4. 임대인이 바뀌더라도 계약은 승계되지 않는다. ",
        "5. 집 주인이 바뀌면 임대차 계약은 해지된다. ",
        "6. 새로운 집 주인이 임대차 계약 해지를 원하는 경우 이 계약은 만료된다."
    ],
    [
        "1. 월세를 세번 이상 밀린 경우, 집 주인이 전기와 수도를 끊을 수 있으며, 임차인은 이에 동의한다.",
        "2. 월세를 밀린 경우 단전 조치 할 수 있다.",
        "3. 월세를 밀린 경우 단수 조치 할 수 있다.",
        "4. 연체 시 단전·단수할 수 있다",
        "5. 차임 2개월 이상 연체하면 단전·단수가 가능하다."
    ],
    [
        "1. 임차인은 보증금 감액을 요구 할 수 없다.",
        "2. 재계약을 하더라도 보증금은 감액 할 수 없다.",
        "3. 세입자는 보증금 감액을 청구하지 않는다.",
        "4. 보증금을 줄일 수 없으므로 보증금 감액을 요구하지 않는다.",
        "5. 보증금 증감청구는 인정되지 않는다. "
    ],
    [
        "1. 이 계약의 묵시적 갱신은 성립하지 않는다.",
        "2. 임차인은 계약이 만료되면 무조건 이사한다.",
        "3. 이 계약은 2년 뒤 해지되며, 연장되지 않는다.",
        "4. 임차인은 재계약을 요구하지 않는다.",
        "5. 계약이 끝나면 무조건 이사 해야한다."
    ],
    [
        "1. 세입자가 구해지면 보증금을 반환한다.",
        "2. 세입자가 구해질 때 까지 보증금을 반환하지 못한다.",
        "3. 다음 임차인이 구해지면 보증금을 반환하도록 하겠다.",
        "4. 다음 임차인이 구해질 때 까지 보증금을 반환하지 않는다.",
        "5. 보증금은 다음 세입자가 구해진 뒤에 반환한다.",
        "6. 보증금은 다음 임차인이 구해진 뒤 반환 가능하다."
    ],
    [
        "1. 신규 임차인이 확보되기 전까지 임대료와 관리비 전액을 임차인이 부담한다",
        "2. 새 임차인을 구하기 전까지는 기존 임차인이 임대료를 부담한다.",
        "3. 새 임차인을 구하기 전까지는 기존 임차인이 관리비를 부담한다.",
        "4. 임대료를 3회 이상 납입하지 않는경우, 새 임차인을 구하기 전까지는 기존 임차인이 관리비를 부담한다.",
        "5. 세입자 과실로 계약이 해지되는 경우, 다음 세입자를 구하기 전까지 발생하는 임대료와 관리비를 기존 세입자가 부담한다."
    ],
    [
        "1. 월세를 한번이라도 연체한 경우 계약을 해지한다.",
        "2. 차임을 한번 이상 연체하면 계약을 해지한다. ",
        "3. 임차인의 차임연체액이 2기의 차임액에 달하는 때에는 임대인은 계약을 해지할 수 있다.",
        "4. 차임을 연속 두번 연체한 경우 계약은 해지된다.",
        "5. 월세를 연속 두 번 연체하면 계약은 해지된다.",
        "6. 월세를 연속 두 번 미납하면 계약을 해지한다.",
        "7. 차임을 연속 두번 미납하면 계약을 해지한다."
    ],
    [
        "1. 임차인은 차임 감액을 요구하지 않는다.",
        "2. 임차인은 월세 감액을 요구하지 않는다.",
        "3. 재계약 시 월세 감액은 없는 것으로 간주한다.",
        "4. 계약 연장 시 차임 감액은 없는 것으로 간주한다.",
        "5. 월세 감액 요구를 하지 않는다.",
        "6. 차임 감액 청구 금지"
    ],
    [
        "1. 계약갱신 요구권을 행사하지 않는다.",
        "2. 이 계약에서 계약 갱신 요구권을 인정하지 않는다.",
        "3. 계약 갱신을 요구하지 않는다.",
        "4. 계약 연장은 불가능하다.",
        "5. 이 계약은 어떤 상황에서도 연장되지 않는다."
    ],
    [
        "1. 묵시적 갱신으로 계약이 연장 된 경우, 세입자는 무조건 2년간 살아야한다.",
        "2. 계약의 묵시적 갱신 시점부터 세임자는 2년간 입주 상태를 유지해야한다.",
        "3. 묵시적 갱신으로 계약이 연장된 경우, 계약은 2년간 유지된다.",
        "4. 묵시적 갱신으로 계약이 연장된 경우, 2년간 계약을 해지할 수 없다.",
        "5. 계약 묵시적 갱신부터 계약은 2년간 유지되며, 2년이 지나기 전에는 계약을 해지 할 수 없다."
    ],
    [
        "1. 소음에 대한 항의를 하지 않는다.",
        "2. 벽간 소음은 다른 세입자로 인한 것이므로 임대인에게 책임이 없다.",
        "3. 건물 수리 중 발생하는 소음에 대해 항의하지 않는다.",
        "4. 집 상태를 위한 행위로 발생하는 소음에 대해 임차인은 항의하지 않는다.",
        "5. 건물 상태를 위한 행위로 발생하는 소음에 대해 임차인은 항의하지 않는다."
    ],
    [
        "1. 월세를 연체한 이력이 있다면 계약 갱신을 거절 할 수 있다.",
        "2. 한번이라도 월세를 미납하는 경우 임대인은 계약 갱신을 거절 할 수 있다.",
        "3. 차임을 연체하면 임대인은 계약 갱신을 거절 할 수 있다.",
        "4. 차임을 한번이라도 미납하는 경우 임대인은 임차인의 계약 갱신 요구를 거절 할 수 있다.",
        "5. 임차인이 월세를 밀린 경우 임대인은 계약 갱신을 거절 할 수 있다."
    ],
    [
        "1. 전입 신고를 하지 않는다.",
        "2. 임차인은 전입 신고를 하지 않는다.",
        "3. 계약 이후 전입 신고를 하지 않는다.",
        "4. 임차인은 입주 후 전입 신고를 하지 않는다.",
        "5. 임차인은 입주 후 전입 신고를 하지 않을 것을 약속한다.",
        "6. 전입 신고를 하지 않기로 약속한다.",
        "7. 전입 신고를 할 경우 계약을 파기 한다."
    ],
]

distance_array = []


# ✅ 기존 특약 데이터 배열 벡터화해서 저장하기
for i in range(len(initialData)):
    contents = initialData[i]

    contents_tokens = [t.morphs(row) for row in contents]

    contents_for_vectorize = []
    for content in contents_tokens:
        sentence = ""
        for word in content:
            sentence = sentence + " " + word
        contents_for_vectorize.append(sentence)

    X = vectorizer.fit_transform(contents_for_vectorize)

    num_samples, num_features = X.shape  # csr_matrix의 형태이다.

    mtx_path = "model/{}model.mtx".format(i)
    tf_path = "model/{}tf.pickle".format(i)

    with open(mtx_path, "wb") as fw:
        pickle.dump(X, fw)

    with open(tf_path, "wb") as fw:
        pickle.dump(vectorizer, fw)


def get_best(case_num, input):
    # 기존 X 뽑아 오는 부분

    mtx_path = "model/{}model.mtx".format(case_num)
    tf_path = "model/{}tf.pickle".format(case_num)

    with open(mtx_path, "rb") as fr:
        X = pickle.load(fr)

    with open(tf_path, "rb") as fr:
        vectorizer = pickle.load(fr)

    new_post = input  # ✅ 입력으로 들어온 특약 배열
    new_post_tokens = [
        t.morphs(row) for row in new_post
    ]  # [['입력', '된', '문', '장임'], ['두번째', '문장'], ['세번', '째', '분장']]

    new_post_for_vectorize = []

    for content in new_post_tokens:
        sentence = ""
        for token in content:
            sentence += " " + token
        new_post_for_vectorize.append(
            sentence
        )  #  [' 입력 된 문 장임', ' 입력 된 문 장임', ' 입력 된 문 장임']

    new_post_vec = vectorizer.transform(new_post_for_vectorize)  # 출력해도 안보임

    # ✅ 거리 계산

    d = 0

    temp_distance = []

    for i in range(0, len(initialData[case_num])):
        post_vec = X.getrow(i)

        d = (case_num, i, dist_raw(post_vec, new_post_vec))
        temp_distance.append(d)
    
    temp_distance = sorted(temp_distance, key=lambda x: x[2])  # 거리 가까운 순으로 정렬

    min_value = temp_distance[0]

    return min_value


@app.route("/api/nlp", methods=["POST"])
def nlp():
    input = request.get_json()
    contents = input["contents"]
    extraInfo = input["extraInfo"]

    # ✅ in
    answer_in = []
    answer_origin = []

    min_distance = 0

    for i in range(len(contents)):  # input으로 들어온 문장 개수만큼 돌리기
        distance_list = [] # 거리 저장 할 리스트

        for j in range(len(initialData)):  # 보유 중인 case 개수만큼 돌리기
            min_distance = get_best(j, [contents[i]])  # 케이스별 (케이스 번호, 인덱스, 거리)
            distance_list.append(min_distance)
            

        # GPT에게 distance_list[:2] 2개에 대해 진짜 가까운 문장이 있는지 물어보기
        #print("가깝다고 나온 문장들 모음", distance_list)
        distance_list = sorted(distance_list, key=lambda x: x[2])
        ask = distance_list[:2]
        
        print("질문:",ask)

        for g in ask:
            st1 = contents[i]
            st2 = initialData[g[0]][g[1]]
            prompt = f"The following two sentences are special provisions of monthly rent contracts in Korea. Are the two special terms written for similar cases? answer yes or no. 1. {st1} 2. {st2}"
            gpt_answer = chatGPT(prompt)

            if "Yes" in gpt_answer:
                print('yes')
                if answer_origin:
                    if not answer_origin[-1] == st1:
                        answer_origin.append(st1)
                        answer_in.append(g[0])
                else:
                    answer_origin.append(st1)
                    answer_in.append(g[0])

    print("최종 결과", answer_in)

    # ✅ out 포함 안된 것
    answer_out = []

    # 1) 필수인데 안들어간 것 (유효 - 필수만 넣으면 됨)
    essential = [34, 35, 36, 37, 38]
    for es in essential:
        if not es in answer_in:
            answer_out.append(es)

    # 2) condition
    pet = extraInfo["pet"]
    loan = extraInfo["loan"]
    substitute = extraInfo["substitute"]

    if pet:  # 반려 동물
        if not 77 in answer_in:
            answer_out.append(77)
    if loan:  # 전세 대출
        if not 88 in answer_in:
            answer_out.append(88)
    if substitute:  # 대리인
        if not 99 in answer_in:
            answer_out.append(99)

    # ✅ 복비 계산
    monthly = extraInfo["monthly"]  # 월세or전세
    commission = extraInfo["commission"]  # 복비
    deposit = extraInfo["deposit"]  # 월세 - 보증금
    monthlyMoney = extraInfo["monthlyMoney"]  # 월세
    lumpSumMoney = extraInfo["lumpSumMoney"]  # 전세금

    scale = 0
    rate = 0
    limit = 0
    answer_commission = 0
    is_expensive = False

    # 1) 거래금액 Scale 계산하기
    if monthly:  # 월세
        if deposit + monthlyMoney * 100 <= 50000000:  # 5천만원 이하면
            scale = deposit + monthlyMoney * 70
        else:
            scale = deposit + monthlyMoney * 100
    else:  # 전세
        scale = lumpSumMoney

    # 2) 거래금액에 따른 상한요율 rate, 한도액 limit 계산
    if scale < 50000000:  # 5천만원 미만 / 0.5 (rate) / 20만(limit)
        rate = 0.005
        limit = 200000
    elif scale < 100000000:  # 5천 이상, 1억 미만 / 0.4 / 30만
        rate = 0.004
        limit = 300000
    elif scale < 600000000:  # 1억 이상, 6억 미만 / 0.3 / 없음
        rate = 0.003
        limit = float("inf")
    elif scale < 1200000000:  # 6억 이상, 12억 미만 / 0.4 / 없음
        rate = 0.004
        limit = float("inf")
    elif scale < 1500000000:  # # 1억 이상, 6억 미만 / 0.5 / 없음
        rate = 0.005
        limit = float("inf")
    else:  # 1억 이상, 6억 미만 / 0.6 / 없음
        rate = 0.006
        limit = float("inf")

    # 3) 최대 복비 계산
    answer_commission = scale * rate
    if answer_commission > limit:
        answer_commission = limit

    # 4) 바가지 당첨
    if answer_commission < commission:
        is_expensive = True


    return jsonify(
        {
            "in": answer_in,
            "out": answer_out,
            "answer_commission": answer_commission,
            "is_expensive": is_expensive,
            "answer_origin":answer_origin,
            "original":contents,
          
        }
    )

@app.route("/api/summary", methods=["POST"])
def summary():
    print(request)
    input = request.get_json()
    contents = input["contents"]
    
    gpt_answer = []

    for i in range(len(contents)):
        prompt = f"다음 내용을 요약해라. 최대한 짧게, 핵심만 담아서, 아주 친절하고 이해하기 쉽게 다시 작성하라. content: {contents[i]}"
        gpt_answer.append(chatGPT(prompt)) 

    return jsonify(
        {
         "summarys":gpt_answer
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
