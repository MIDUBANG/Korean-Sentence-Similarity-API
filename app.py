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
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

