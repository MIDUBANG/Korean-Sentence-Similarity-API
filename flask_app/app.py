# -*- coding: utf-8 -*-
import openai
import os
from flask import Flask, render_template
from flask import request, jsonify
from konlpy.tag import Okt
from flask_cors import CORS
from PIL import Image
from botocore.exceptions import ClientError
from konlpy.tag import Okt
import scipy as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import jsonpickle
import requests
import time
import logging
import uuid
import boto3
from dotenv import load_dotenv, find_dotenv


from data import initialDataSet

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

API_KEY= os.getenv("FLASK_API_KEY")

CORS(app)
load_dotenv()
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_KEY= os.environ.get("AWS_SECRET_KEY")
X_OCR_SECRET= os.environ.get("X_OCR_SECRET")

S3_LOCATION = f"http://midubang-s3.s3.amazonaws.com/"


def crossCheckingGPT(st1, st2, API_KE=API_KEY):
    # set api key
    openai.api_key = API_KEY
    # Call the chat GPT API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
         {"role": "system", "content": "You are an LLM (Language Model) that, upon receiving two sentences as input, determines whether they have the same semantic meaning and responds with either 'Yes.' or 'No.'."},
         {"role": "user", "content": f"Q: For the sentence pair '계약 연장은 계약 만기 3달 전까지 갱신 의사를 밝혀야만 가능하다.' and '계약 만기 5개월 전 까지 재계약 의사를 밝히지 않은 경우, 계약은 만료되는 것으로 간주한다.', do these two sentences have the same semantics?"},
         {"role": "assistant", "content": "A: First, identify the key differences between the two sentences. Second, consider the impact of the difference in wording. Third, consider the overall meaning of the two sentences. Therefore, given that the two sentences convey the same general idea, despite the difference in wording, we can conclude that they have the same semantics. The answer (yes or no) is: yes."},
         {"role":"user", "content": f"Q: For the sentence pair '임대인이 사전에 고지하지 않은 체납 사실이 확인된 경우에는 계약을 해지하며, 임차인에게 계약금을 돌려준다.' and '계약 시 임대인이 임차인에게 국세,지방세 체납이나 근저당권 이자 체납이 있는지 알리고, 계약 체결 후에는 임대인이 세무서, 지방자치 등에 이를 확인할 수 있게 한다. 고지하지 않은 체납 사실이 확인되면 계약을 해지한다.', do these two sentences have the same semantics?"},
         {"role": "assistant", "content": "A: First, identify the key differences between the two sentences. Second, consider the impact of the difference in wording. Third, consider the overall meaning of the two sentences. Therefore, Both sentences share the same semantic meaning, which is that the landlord must inform the tenant in advance about the nonpayment of taxes, and if not done so, terminate the contract. Therefore, we can conclude that the two sentences have the same semantic. The answer (yes or no) is: yes."},
         {"role": "user", "content": f"Q: For the sentence pair '다음 임차인이 구해지면 보증금을 반환한다.' and '세입자가 구해질 때 까지 보증금을 반환하지 못한다.', do these two sentences have the same semantics?"},
         {"role": "assistant", "content": "A: First, identify the key differences between the two sentences. Second, consider the impact of the difference in wording. Third, consider the overall meaning of the two sentences. Therefore, Both sentences share the same semantic meaning, which is that the security deposit will not be refunded until the next tenant is found, but it will be returned once the next tenant is secured. Therefore, we can conclude that the two sentences have the same semantics. The answer (yes or no) is: yes."},
         {"role": "user", "content": f"Q: For the sentence pair {st1} and {st2}, do these two sentences have the same semantics? The answer (yes or no) is: ____"}
        ],
        temperature=0,
    )
    return completion["choices"][0]["message"]["content"].encode("utf-8").decode()

def crossCheckingWithoutCoT(st1, st2, API_KE=API_KEY):
    # set api key
    openai.api_key = API_KEY
    # Call the chat GPT API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
         {"role": "user", "content": f"Q: For the sentence pair '계약 연장은 계약 만기 3달 전까지 갱신 의사를 밝혀야만 가능하다.' and '계약 만기 5개월 전 까지 재계약 의사를 밝히지 않은 경우, 계약은 만료되는 것으로 간주한다.', do these two sentences have the same semantics?"},
        {"role": "user", "content": f"Q: For the sentence pair {st1} and {st2}, do these two sentences have the same semantics? The answer (yes or no) is: ____"}
        ],
        temperature=0,
    )
    return completion["choices"][0]["message"]["content"].encode("utf-8").decode()


def summaryGPT(prompt):
    # set api key
    openai.api_key = API_KEY
    # Call the chat GPT API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
         {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return completion["choices"][0]["message"]["content"].encode("utf-8").decode()




def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


t = Okt()

vectorizer = TfidfVectorizer(min_df=1, decode_error="ignore")


# case 순서대로 쭉 나열
initialData = initialDataSet

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

# @app.route("/api/gpttest", methods=["POST"])
# def gpt_test():
#     input = request.get_json()
#     st1 = input['st1']
#     st2 = input['st2']
#     gpt_answer = crossCheckingGPT(st1,st2,API_KEY)
#     gpt_answer2 = crossCheckingWithoutCoT(st1, st2, API_KEY)
    
#     print("------------------------------------------")
#     print("CoT를 적용하지 않은 프롬프팅")
#     print("------------------------------------------")

#     print('문장1:', st1)
#     print('문장2:', st2)
#     print('답변:',gpt_answer2)

#     print("------------------------------------------")
#     print("CoT를 적용한 프롬프팅")
#     print("------------------------------------------")
#     print('문장1:', st1)
#     print('문장2:', st2)
#     print('답변:',gpt_answer)

#     return jsonify({
#         'gpt_answer' : gpt_answer
#     })



@app.route("/api/nlp", methods=["POST"])
def nlp():
    chatapi_request_num = 0

    input = request.get_json()
    contents = input["contents"]
    extraInfo = input["extraInfo"]

    # ✅ in
    answer_in = []
    answer_origin = []

    min_distance = 0

    for i in range(len(contents)):  # input으로 들어온 문장 개수만큼 돌리기
        distance_list = []  # 거리 저장 할 리스트

        for j in range(len(initialData)):  # 보유 중인 case 개수만큼 돌리기
            min_distance = get_best(j, [contents[i]])  # 케이스별 (케이스 번호, 인덱스, 거리)
            distance_list.append(min_distance)

        # GPT에게 distance_list[:2] 2개에 대해 진짜 가까운 문장이 있는지 물어보기
        distance_list = sorted(distance_list, key=lambda x: x[2])
        ask = distance_list[:2]

        print("질문:", ask)


        for g in ask:
            st1 = contents[i]
            st2 = initialData[g[0]][g[1]]
            gpt_answer = crossCheckingGPT(st1,st2)

            print("비교 대상 문장 : ", st2)
            print('답변:',gpt_answer)

            if "Yes" in gpt_answer or "yes" in gpt_answer:
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
    essential = [32, 33, 34, 35, 36, 37, 38,39,40,41,42,43,45,46,47,48,49,50,51,52]
    for es in essential:
        if not es in answer_in:
            answer_out.append(es)

    # 2) condition
    pet = extraInfo["pet"]
    loan = extraInfo["loan"]
    substitute = extraInfo["substitute"]

    if pet:  # 반려 동물
        if not (25 in answer_in or 25 in answer_out):
            answer_out.append(25)
    if loan:  # 전세 대출
        if not (36 in answer_in or 36 in answer_out):
            answer_out.append(36)
        if not (46 in answer_in or 46 in answer_out):
            answer_out.append(46)
    if substitute:  # 대리인
        if not (52 in answer_in or 52 in answer_out):
            answer_out.append(52)

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
            "answer_origin": answer_origin,
            "original": contents,
        }
    )


@app.route("/api/summary", methods=["POST"])
def summary():
    print(request)
    input = request.get_json()
    contents = input["contents"]

    gpt_answer = []

    for i in range(len(contents)):
        # prompt = f"You are a report analysis robot. Summarize [content]. 1. Summarize as concisely as possible. 2. Summarize with only the key points. 3. Write in a very friendly and understandable tone. 4. Write in respectful language. 5.Translate into Korean (does not print English results) 6.Get rid of all the extraneous words, and stick to the main points. => content: {contents[i]}"
        prompt = f"너는 레포트 요약 로봇이다. content를 요약한 결과를 출력하라. (제한 사항 : 1. 반드시 존댓말로 작성하라. 2. 친절한 말투로 작성하라. 3.핵심 내용만 담아라. 4. 최대한 짧게 요약하라.) content: {contents[i]}"
        gpt_answer.append(summaryGPT(prompt))

    return jsonify({"summarys": gpt_answer})




@app.route("/api/message", methods=["POST"])
def TextMassageMaker(API_KEY=API_KEY):
    input = request.get_json()
    receiver = input["receiver"]
    purpose = input["purpose"]
    tone = input["tone"]
    more_info = input["more_info"]

    prompt = f"조건에 맞게 문자 메세지를 작성하라. 최대한 길게 작성하라. \n 1. 수신자 : ${receiver}\n2. 문자 쓰는 목적 : ${purpose}\n3. 문자의 어조 : ${tone}\n4. 추가적인 상황 정보 : ${more_info}\n\n 조건\n- 문자의 시작은 '안녕하세요'로 한다\n- 도입에 내가 누구인지 밝힌다.\n- 서론, 본론, 결론의 구성으로 작성하고 문단별로 줄바꿈을한다다.\n- 문자 내용만 출력한다."
    # set api key
    openai.api_key = API_KEY
    # Call the chat GPT API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )

    message_result = completion["choices"][0]["message"]["content"].encode("utf-8").decode()

    return jsonify({"result": message_result})


@app.route("/api/ocr", methods=["POST"])
def clovaocr_from_image():
    load_dotenv(find_dotenv())
    image = request.files['image']
    user_id = request.form['id']
    
    #이미지 저장
    im = Image.open(image)
    path, ext = os.path.splitext(image.filename)
    imgpath = f'uploads/ocr_image{ext}'
    im.save(imgpath) # 파일명을 보호하기위한 함수, 지정된 경로에 파일 저장

    # s3 업로드
    s3_url = upload_image(image, imgpath, user_id)

    headers = {
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'X-OCR-SECRET' :X_OCR_SECRET

    }

    requestJson = {
            "images": [{
                "format": "png",
                "name": "medium",
                "data": None,
                "url": s3_url,
            }],
            "lang": "ko",
            "requestId": "string",
            "resultType": "string",
            "timestamp": int(round(time.time() * 1000)),
            "version": "V1"
    }

    clova_url =  "https://g762ivic4j.apigw.ntruss.com/custom/v1/22201/7c50f467463caa6b09d883d04208f2266ca3abf653480b4b0f93a460562499c6/general"

    res = requests.post(clova_url, json=requestJson, headers=headers)
    result = res.json()
    infer_texts = [field["inferText"] for field in result["images"][0]["fields"]]
    processsed_texts = get_cases(infer_texts)
    data = {'text': processsed_texts, 's3_url':s3_url}
    return jsonify(data)
    # return jsonpickle.encode()

def get_cases(inputlist):
    # 특약사항 부분만 추출
    start_index = inputlist.index('특약사항')
    last_index = len(inputlist) - 1 - inputlist[::-1].index('본')
    output = inputlist[start_index+1:last_index]

    #추출된 원소들을 문장으로 조합
    sentences = []
    sentence = ""
    for element in output:
        sentence += element
        if element.endswith("다."):
            sentences.append(sentence)
            sentence = ""
        else:
            sentence += " "
    if sentence:
        sentences.append(sentence)

    return sentences



def upload_image(image, imgpath, user_id):
    try:
        
        # filename = secure_filename(image.filename)
        image.filename = get_unique_imgname(image.filename, user_id)
    
        s3 = s3_connection()
        # s3.put_object(Bucket=BUCKET_NAME, Body=image, Key=image.filename)
        s3.upload_file(imgpath, "midubang-s3", image.filename,ExtraArgs={
                "ACL": "public-read",
                "ContentType": image.content_type
            } )
    except ClientError as e:
        logging.error(e)
        return None
    
    url = f"{S3_LOCATION}{image.filename}"
    
    return url    

def get_unique_imgname(filename, user_id):
    ext = filename.rsplit(".",1)[1].lower()
    # 이미지 파일명 구조 : {userid}/{uuid}.{확장자} 
    # 이미지의 원래 이름에 s3 파일이름에 들어갈 수 없는 문자가 있을 경우를 고려하여 일단 uuid만 넣었음
    return f"{user_id}/{uuid.uuid4().hex}.{ext}"  


def s3_connection():
    s3 = boto3.client('s3',aws_access_key_id = AWS_ACCESS_KEY, aws_secret_access_key = AWS_SECRET_KEY)
    return s3


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



# Are the two special terms written for similar cases? answer yes or no. 1. {st1} 2. {st2}
#         {"role": "system", "content": "Even if two sentences are included in each other, they are judged to have similar meanings."},
# Are the two special terms written for similar cases?
# def testGPT(st1, st2, API_KEY=YOUR_API_KEY):
#     # set api key
#     openai.api_key = API_KEY
#     # Call the chat GPT API
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#         {"role": "system", "content": "You are a machine that determines whether two sentences are similar."},
#         {"role": "system", "content": "The following two sentences are special provisions of monthly rent contracts in Korea."},
#         {"role": "system", "content": "Determine if both special contracts are written for similar cases."},
#         {"role": "system", "content": "If the purpose of the two clauses is the same, if one sentence includes the other sentence, or if the core meaning of the two sentences is the same, the evaluation is 'yes'."},
#         {"role": "system", "content": "The answer format should be yes or no only."},
#         {"role": "user", "content": f"answer yes or no =>  s1. {st1} 2. {st2}"}
#         ],
#         temperature=0,
#         max_tokens=10,
#     )
#     return completion["choices"][0]["message"]["content"].encode("utf-8").decode()



# # temperature=0,
# st1 = "임차인은 월세 감액을 요구하지 않는다."
# st2 = "재계약 시 월세 감액은 없는 것으로 간주한다."
# result = testGPT(st1, st2) # system 메세지와
# print(result)

# print("2실행")
# prompt = f'The following sentence "input" is one of the special provisions of the monthly rent contract in Korea. And the initialdata array is a case-by-case grouping of monthly rent contract terms that have a similar context. When adding an input sentence to an initial data array, answer the index location that needs to be inserted in the initial data[n] format. input = "세입자는 계약이 끝날 때 까지 전입 신고를 하지 않을 것을 약조한다. " initialData = [["월세와 보증금은 매년 시세에 맞게 조정할 수 있다.","시세에 맞게 월세를 인상한다.","시세에 맞게 보증금을 인상한다.","금리 인상에 맞추어 월세를 인상한다.","계약을 연장할 경우 월세를 인상한다."],....["1. 전입 신고를 하지 않는다.","2. 임차인은 전입 신고를 하지 않는다.","3. 계약 이후 전입 신고를 하지 않는다.","4. 임차인은 입주 후 전입 신고를 하지 않는다.","5. 임차인은 입주 후 전입 신고를 하지 않을 것을 약속한다.","6. 전입 신고를 하지 않기로 약속한다.","7. 전입 신고를 할 경우 계약을 파기 한다."],]'
# print(chatGPT(prompt))


# import time


# print('====== 1 실행=====')
# record1 = []
# answer_count1 = 0
# for i in range(len(testData)):
#     print(i,"/",len(testData))
#     time.sleep(1)
#     result = testGPT(testData[i][0],testData[i][-1])
    
#     if 'yes' in result or 'Yes' in result:
#         answer_count1 += 1
#     else:
#         record1.append(i) 

# time.sleep(3)

# print('<<<<<<<< 1번 API 실행 결과 <<<<<<<< ')
# print("정답 개수 : ",answer_count1)
# print("오답 개수 : ",len(testData) - answer_count1)
# print("오답이 나온 케이스 >> \n", record1)


# print('====== 2 실행=====')
# record2 = []
# answer_count2 = 0
# for i in range(len(testData)):
#     print(i,"/",len(testData))
#     time.sleep(2)
#     prompt = f"The following two sentences are special provisions of monthly rent contracts in Korea. Are the two special terms written for similar cases? answer yes or no. 1. {testData[i][0]} 2. {testData[i][-1]}"
#     result = chatGPT(prompt)

#     if 'Yes' in result:
#         answer_count2 += 1
    
#     record2.append([i,result])

# time.sleep(3)

# print('<<<<<<<< 2번 API 실행 결과 <<<<<<<< ')
# print("정답 개수 : ",answer_count2)
# print("오답 개수 : ",len(testData) - answer_count2)
# print("상세 결과 >> \n", record2)
