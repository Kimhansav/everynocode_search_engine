import numpy as np
import os
import pandas as pd
import urllib.request
import faiss
import time
import json
import ast
import torch
from functools import wraps
from flask import Flask, request, jsonify, make_response, Response
from transformers import AutoModel, AutoTokenizer
from google.cloud import storage
from datetime import datetime, timedelta, timezone

model = None
tokenizer = None
index = None
data = None

def create_app():
    app = Flask(__name__)
    # app.config['JSON_AS_ASCII'] = False

    with app.app_context():
        init_db()
    
    @app.route('/search', methods=['GET'])
    def find_similar():
        user_input = request.args.get('q')
        # 유사도 계산 및 가장 유사한 텍스트 찾기 (여기서는 로직 구현 필요)
        results = search(user_input)
        return jsonify(results)
    return app

#모델 경로를 지정
HUGGINGFACE_MODEL_PATH = 'BM-K/KoSimCSE-bert-multitask'

# #json 데이터를 리턴할 때 한글이 깨지는 현상을 막기 위한 데코레이터
# def as_json(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         res = f(*args, **kwargs)
#         res = json.dumps(res, ensure_ascii = False).encode('utf8')
#         return Response(res, content_type = 'application/json; charset = utf-8')
#     return decorated_function

#Google cloud storage에서 파일을 다운로드하는 함수
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """GCS에서 파일을 다운로드하고 로컬 파일 시스템에 저장합니다."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(f"{source_blob_name} downloaded to {destination_file_name}.")

#검색 함수에서 사용할 사용자 입력 인코딩 함수
def encode(query):
  try:
    inputs = tokenizer(query, padding = True, truncation = True, return_tensors = 'pt')

    with torch.no_grad():
            outputs = model(**inputs)

    #배치 단위가 아니라 단일 문장이기 때문에 mean pooling을 할 때 dim = 0으로 설정
    mean_pooling = torch.mean(outputs.last_hidden_state, dim = 0)
    return mean_pooling
  except Exception as e:
      app.logger.error(f'Error in encode: {str(e)}')
      raise e

#검색 함수
def search(query):
  try:
    t = time.time()
    query_vector = encode(query)
    k = 5
    top_k = index.search(query_vector, k)
    print('total time: {}'.format(time.time() - t))
    top_k_ids = top_k[1].tolist()[0]
    #bubble.io의 api connector가 데이터를 json으로 읽을 수 있도록 형식 변경
    # results = {}
    # top_k_ids = top_k[1].tolist()[0]
    # for i, _id in enumerate(top_k_ids):
    #     #url이 비어 있으면(카톡 대화문이면) check = True
    #     check = True if pd.isna(data.iloc[_id]['url']) == True else False
    #     results[f'{i} question'] = data.iloc[_id]['question']
    #     results[f'{i} answer'] = ' '.join(s for s in ast.literal_eval(data.iloc[_id]['answer'])) if check == True else data.iloc[_id]['answer']
    #     results[f'{i} question summary'] = data.iloc[_id]['question_summary']
    #     results[f'{i} answer summary'] = data.iloc[_id]['answer_summary']
    #     results[f'{i} question with url'] = data.iloc[_id]['question_withurl']
    #     results[f'{i} answer with url'] = ' '.join(s for s in ast.literal_eval(data.iloc[_id]['answer_withurl'])) if check == True else data.iloc[_id]['answer_withurl']
    #     results[f'{i} questioner'] = data.iloc[_id]['questioner']
    #     results[f'{i} respondent'] = ' '.join(s for s in ast.literal_eval(data.iloc[_id]['respondent'])) if check == True else '.'
    #     results[f'{i} date'] = data.iloc[_id]['date']
    #     results[f'{i} url'] = '.' if check == True else data.iloc[_id]['url']

    # return results
    return {i : {'question' : data.iloc[_id]['question'], 'answer' : data.iloc[_id]['answer'], 'question_summary' : data.iloc[_id]['question_summary'], 'answer_summary' : data.iloc[_id]['answer_summary'], 'question_withurl' : data.iloc[_id]['question_withurl'], 'answer_withurl' : data.iloc[_id]['answer_withurl'], 'questioner' : data.iloc[_id]['questioner'], 'respondent' : data.iloc[_id]['respondent'], 'date' : data.iloc[_id]['date'], 'url' : data.iloc[_id]['url']} for i, _id in enumerate(top_k_ids)}
  except Exception as e:
      app.logger.error(f'Error in search: {str(e)}')
      raise e
#초기 호출 시 모델과 토크나이저를 전역 변수로 서버에 캐싱하는 함수
def load_model_tokenizer():
    model = AutoModel.from_pretrained(HUGGINGFACE_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
    return model, tokenizer

#초기 호출 시 csv 데이터를 데이터프레임으로 변형, 데이터프레임과 FAISS 인덱스를 전역 변수로 서버에 캐싱하는 함수
def load_df_index():
    #데이터 불러오기
    download_blob('bubble_project', 'sentence_embed_result_short.csv', '/tmp/sentence_embed_result_short.csv')
    df = pd.read_csv('/tmp/sentence_embed_result_short.csv')
    # df = pd.read_csv('sentence_embed_result_short.csv')
    #csv파일로 저장할 때 쉼표를 살리기 위해 json.dumps로 저장했기 때문에 json.loads 사용
    df['embedding'] = df['embedding'].apply(json.loads)
    encoded_data = torch.tensor(df['embedding'].tolist())

    #속도를 위해 임베딩 열을 제외한 데이터프레임을 설정
    #인덱싱 속도를 위해서 데이터프레임을 리스트로 변환
    data = df.drop(['embedding'], axis = 1)

    # FAISS 인덱스 구축
    index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
    index.add_with_ids(encoded_data, np.array(range(0, len(data))))
    faiss.write_index(index, 'question-answer')
    return data, index


def init_db():
    global model, tokenizer, index, data
    if model is None or tokenizer is None or index is None or data is None:
        model, tokenizer = load_model_tokenizer() # 모델, 토크나이저 로딩 함수
        data, index = load_df_index() #데이터프레임, 인덱스 로딩 함수

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
