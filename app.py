#import os
#from sklearn.feature_extraction.text import CountVectorizer
#import konlpy

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

CORS(app)


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
    ["입주 후 생긴 하자는 세입자가 수리한다.", "하자는 임차인이 수리한다.", "하자는 임차인이 책임지고 수리"],
    ["월세를 밀리면 퇴거", "이유 없이 월세를 두 달 이상 밀리면 퇴실", "월세를 내지 못한 경우 퇴실한다."],
    ["반려동물을 키우지 않는다. 금지", "강아지를 키우면 벽지 도배비를 지불한다.", "고양이를 키우지 않는다."],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
    [
        "유독음식물공급·포로에 관한 죄중 법률이 정한 경우에 한하여 단심으로 할 수 있다. 다만, 사형을 선고한 경우에는 그러하지 아니하다.",
        "법률은 특별한 규정이 없는 한 공포한 날로부터 20일을 경과함으로써 ",
        "무원의 범죄나 군사에 관한 간첩죄의 경우와 초병·초소·",
    ],
]
distance_array = []


for i in range(len(initialData)):
    contents = initialData[i]  # ✅ 기존 특약 데이터 배열

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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    contents = initialData[case_num] # ✅ 기존 특약 데이터 배열

    contents_tokens = [t.morphs(row) for row in contents]

    contents_for_vectorize = []
    for content in contents_tokens:
        sentence = ''
        for word in content:
            sentence = sentence + ' ' + word
        contents_for_vectorize.append(sentence)

    X = vectorizer.fit_transform(contents_for_vectorize)

    num_samples, num_features = X.shape # csr_matrix의 형태이다.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 기존 X 뽑아 오는 부분

    mtx_path = "model/{}model.mtx".format(case_num)
    tf_path = "model/{}tf.pickle".format(case_num)

    with open(mtx_path, "rb") as fr:
        X = pickle.load(fr)

    with open(tf_path, "rb") as fr:
        vectorizer = pickle.load(fr)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

    data = np.array(temp_distance)
    min_val = np.min(data)
    print(case_num, "번 케이스와의 최소 거리", min_val)
    return min_val


@app.route("/api/nlp", methods=["POST"])
def nlp():
    input = request.get_json()
    contents = input["contents"]
    extraInfo = input["extraInfo"]

    print(contents)
    print(extraInfo)

    # ✅ in
    min_dis = 1000
    best_case_i = 999
    answer_in = []

    for i in range(len(contents)):  # input으로 들어온 문장 개수만큼 돌리기
        for j in range(len(initialData)):  # 보유 중인 case 개수만큼 돌리기

            sum_of_distance = get_best(j, [contents[i]])  # 인풋 중 하나만

            if min_dis > sum_of_distance:
                min_dis = sum_of_distance
                best_case_i = j

        print(best_case_i)

        if min_dis < 0.7:  # 거리 기준
            answer_in.append(best_case_i)

        min_dis = 1000

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
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
