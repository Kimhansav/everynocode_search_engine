import numpy as np
import os
import pandas as pd
import urllib.request
import faiss
import time
import json
import ast
import torch
import logging
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
from google.cloud import storage
from datetime import datetime, timedelta, timezone

model = None
tokenizer = None
index = None
data = None

#로컬 테스트를 위한 signed_url 생성기
def generate_signed_url(bucket_name, blob_name):
    """Generate a signed URL for the given bucket and blob that expires in one hour."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        expiration=datetime.now(timezone.utc) + timedelta(hours=1),
        method='GET' 
    )
    return url

def create_app():
    app = Flask(__name__)
    logging.basicConfig(level = logging.DEBUG)

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
signed_url = generate_signed_url('bubble_project', 'sentence_embed_result_short.csv')

# #Google cloud storage에서 파일을 다운로드하는 함수
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     """GCS에서 파일을 다운로드하고 로컬 파일 시스템에 저장합니다."""
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     blob.download_to_filename(destination_file_name)

#     print(f"{source_blob_name} downloaded to {destination_file_name}.")

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
    k = 20
    top_k = index.search(query_vector, k)
    print('total time: {}'.format(time.time() - t))

    return [{'question' : data.iloc[_id]['question'], 'answer' : data.iloc[_id]['answer'], 'question_summary' : data.iloc[_id]['question_summary'], 'answer_summary' : data.iloc[_id]['answer_summary'], 'question_withurl' : data.iloc[_id]['question_withurl'], 'answer_withurl' : data.iloc[_id]['answer_withurl'], 'questioner' : data.iloc[_id]['questioner'], 'respondent' : data.iloc[_id]['respondent'], 'date' : data.iloc[_id]['date'], 'url' : data.iloc[_id]['url']} for _id in top_k[1].tolist()[0]]
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
    # download_blob('bubble_project', 'data/sentence_embed_result.csv', '/tmp/sentence_embed_result.csv')
    df = pd.read_csv(signed_url)
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

# @app.route('/search', methods=['GET'])
# def find_similar():
#     user_input = request.args.get('q')
#     # 유사도 계산 및 가장 유사한 텍스트 찾기 (여기서는 로직 구현 필요)
#     results = search(user_input)
#     return jsonify(results)

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
