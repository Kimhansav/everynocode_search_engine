# -*- coding: utf-8 -*-
"""BP_sentence_embed.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1SATXhHBy5-o7D7-4mhELQ1GBOnq-No1A
"""

#질답세션의 임베딩 벡터를 추가한 뒤 .csv로 저장하는 코드(로컬)

!pip install -U sentence-transformers #서버 업로드 시 삭제
!pip install datasets

import numpy as np
import os
import pandas as pd
import urllib.request
import time
import torch
import json
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from google.colab import drive
from transformers import AutoModel, AutoTokenizer, shape_list, TFBertModel, RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline, pipeline, BertTokenizer, BertForNextSentencePrediction,  TrainingArguments, BertForMaskedLM, Trainer, TrainerCallback
from datasets import Dataset, load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
drive.mount('/content/drive')

#.csv파일 불러오기
file_path = '/content/drive/My Drive/summary_result_short.csv'
df = pd.read_csv(file_path)

#데이터프레임의 question 열을 리스트로 변형
question_data = df['question'].to_list()
answer_data = df['answer'].to_list()

print(len(question_data))
print(len(answer_data))

# #모델 불러오기
# model_path = '/content/drive/My Drive/Pretrained_Model_sentence_embed'

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModel.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

#모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HUGGINGFACE_MODEL_PATH = 'BM-K/KoSimCSE-bert-multitask'

model = AutoModel.from_pretrained(HUGGINGFACE_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
model.to(device)

# #수동 배치처리
# batch_size = 64

# result = []
# for i in tqdm(range(0, len(data), batch_size), desc = "Embedding"):
#     batch_texts = data[i : i + batch_size]
#     inputs = tokenizer(batch_texts, padding = True, truncation = True, return_tensors = 'pt')
#     inputs = {k : v.to(device) for k, v in inputs.items()}
#     #모델 실행
#     with torch.no_grad():
#         outputs = model(**inputs)

#     mean_pooling = torch.mean(outputs.last_hidden_state, dim = 1)
#     result.append(mean_pooling)

def get_embeddings(batch_size, data):
  result = []
  for i in tqdm(range(0, len(data), batch_size), desc = "Embedding"):
      batch_texts = data[i : i + batch_size]
      inputs = tokenizer(batch_texts, padding = True, truncation = True, return_tensors = 'pt')
      inputs = {k : v.to(device) for k, v in inputs.items()}
      #모델 실행
      with torch.no_grad():
          outputs = model(**inputs)

      mean_pooling = torch.mean(outputs.last_hidden_state, dim = 1)
      result.append(mean_pooling)

  return result

# print(len(result))
# print(len(result[0]))
# print(len(result[0][0]))

# print(len(qresult))
# print(sum(len(lists) for lists in qresult))
# print(len(aresult))
# print(len(df))

#데이터프레임에 새로운 열로 추가
qresult = get_embeddings(64, question_data)
aresult = get_embeddings(64, answer_data)
df['question_embedding'] = [sentence.cpu().numpy() for batches in qresult for sentence in batches]
df['answer_embedding'] = [sentence.cpu().numpy() for batches in aresult for sentence in batches]

#비율을 적용한 벡터합
k = 0.8
df['embedding'] = [k * df.iloc[i]['question_embedding'] + (1 - k) * df.iloc[i]['answer_embedding'] for i in range(len(df))]

#필요없는 질문임베딩, 답변임베딩은 제거
df = df.drop(['question_embedding', 'answer_embedding'], axis = 1)

#json으로 저장할 파일은 json.dumps 적용 전에 복사해두기
df_json = df.copy()

#csv파일로 저장할 때 쉼표가 사라지지 않게 해야 함, json.dumps 사용
df['embedding'] = df['embedding'].apply(lambda x: json.dumps(x.tolist()))

#json파일로 저장한 뒤 elasticsearch에 사용하기 위해서는 json.dumps를 사용하지 않아야 함
df_json['embedding'] = df_json['embedding'].apply(lambda x: x.tolist())

print(qresult[:3])
print(aresult[:3])
print(df['embedding'].to_list()[:3])

print(len(df['embedding'][55]))
print(df['embedding'])

print(type(df_json['embedding'][0]))

#.csv 파일로 google drive에 저장 (나중에 google cloud storage에 저장해야 함)
save_path = '/content/drive/My Drive/sentence_embed_result_short.csv'

df.to_csv(save_path)

"""##요구되는 형식에 맞도록 json 파일로 저장"""



print(df.to_json(orient="records", indent = 2, force_ascii = False))

#.json 파일로 google drive에 저장
json_save_path = '/content/drive/My Drive/sentence_embed_result_short.json'

df_json.to_json(json_save_path, orient = 'records', indent = 2, force_ascii = False)