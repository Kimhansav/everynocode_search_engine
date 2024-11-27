# -*- coding: utf-8 -*-
"""BP_judge_answer_KCBERT_nsp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ij1KOfw2UFDB6ufjunuROASXPuWw2JHu
"""

#한국 버블 커뮤니티 오픈톡방 대화의 질문에 대한 답변을 선별하는 코드(BERT Next Sentence Prediction)(로컬)

!pip install soynlp
!pip install datasets
!pip install accelerate -U
import accelerate
import random
import pandas as pd
import numpy as np
import re
import os
import torch
import tensorflow as tf
import urllib.request
from datasets import Dataset, load_dataset, ClassLabel
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, shape_list, TFBertModel, RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline, pipeline, BertTokenizer, BertForNextSentencePrediction,  TrainingArguments, BertForMaskedLM, Trainer, TrainerCallback
from sklearn.model_selection import StratifiedKFold, train_test_split
from google.colab import drive
drive.mount('/content/drive')

#NSP로 훈련된 모델 로드
finetuned_model_load_path = '/content/drive/My Drive/Finetuned_Model_judge_answer'

finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_load_path)
finetuned_model = BertForNextSentencePrediction.from_pretrained(finetuned_model_load_path)

"""# 중요 : DataFrame의 Unnamed: 0을 사용해야 인덱스로 텍스트를 관리할 수 있다."""

#url 삭제하지 않은 데이터셋 로드
withurl_path = '/content/drive/My Drive/talk_preprocess_result_short_urlexist.xlsx'

df_withurl = pd.read_excel(withurl_path)

#열 이름 변경
df_withurl.rename(columns={'Unnamed: 0': 'number'}, inplace=True)

print(df_withurl)

#카카오톡 데이터 불러오기
file_path = '/content/drive/My Drive/judge_question_result_short_KcBERT.csv'

#카카오톡 대화내용을 데이터프레임으로 받기
df = pd.read_csv(file_path)

#Unnamed: 0 열을 사용하기 위해 열 이름 교체
df.rename(columns={'Unnamed: 0': 'number'}, inplace=True)

print(df)

#질문으로 판별된 텍스트를 새 데이터프레임으로 생성
df_question = df[df['label'] == 'question']

print(df_question)

#새 데이터프레임의 text, index, name, date를 질문 딕셔너리에 저장
question_index_creator_date_dict = {text : {'index' : index, 'name' : name, 'date' : date} for (text, index, name, date) in zip(df_question['text'], df_question['number'], df_question['name'], df_question['date'])}
print(question_index_creator_date_dict)

# Ensure using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = finetuned_model.to(device)
tokenizer = finetuned_tokenizer

#중복 질문 제거를 동시에 수행하기 위한 새로운 알고리즘

#실험을 위한 하이퍼파라미터 설정
#후보군으로 삼을 텍스트 개수 범위
text_range = 20
#답변 목록에 추가할지 기준이 되는 레이블값
standard = 0.5

#질문-답변 쌍 딕셔너리 생성. index로 질문 번호를 관리하기 때문에 value를 리스트로 작성하고, 리스트 안에는 {'question' : 질문내용, 'answer' : 답변내용, 'question_withurl' : url 포함한 질문내용, 'answer_withurl' : url 포함한 답변내용, 'questioner' : 작성자명, 'respondent' : 답변자 목록, 'date' : 작성날짜} 가 포함되도록 한다.
qa_pair_dictionary = {info['index'] : {'question' : question, 'answer' : [], 'question_withurl' : df_withurl[df_withurl['number'] == info['index']]['text'].iloc[0], 'answer_withurl' : [], 'questioner' : info['name'], 'respondent' : [], 'date' : info['date']} for (question, info) in question_index_creator_date_dict.items()}

#질문 속 질문인지 판별할 때 사용할 불리언
in_question_texts = False

all_texts = {index : {'text' : text, 'name' : name} for index, text, name in zip(df['number'], df['text'], df['name'])}

for index, item in tqdm(all_texts.items(), desc = 'Processing Answer to Question'):

  candidate_qa_list = [] #현재 텍스트의 소속을 판정할 (질문-답변 딕셔너리) 리스트
  candidate_qa_index_list = []
  in_question_texts = True if item['text'] in question_index_creator_date_dict else False #판별할 텍스트가 질문인지 검사 #df.iloc[index]['label'] == 'question'
  # if in_question_texts == True and len(tokenizer(item['text'])) < 10: #토큰화된 길이가 10 이하인 질문은 단독 질문 목록에서 제거
  #   del qa_pair_dictionary[index]

  start = 0 if index < 20 else index - text_range - 1 #인덱스가 20 미만일 경우 검사 범위 조정
  for i in range(start, index): #현재 텍스트가 소속될 질문의 범위
    candidate = qa_pair_dictionary.get(i, None) #qa_pair_dictionary에서 i 인덱스에 해당하는 질답 딕셔너리 가져오기
    if candidate != None:
      if candidate['question'] not in item['text']: #질문과 답변이 같은 경우 제외
        candidate_qa_list.append(candidate) #결과 리스트에서 최대 확률인 질문을 인덱싱하기 위해 인덱스를 포함한 딕셔너리를 append
        candidate_qa_index_list.append(i)

  if len(candidate_qa_list) == 0:
    continue

  #데이터를 튜플로 묶은 뒤 배치처리
  batched_data = [(qa_dict['question'] + ' '.join(qa_dict['answer']), item['text']) for qa_dict in candidate_qa_list]
  inputs = tokenizer(batched_data, padding = True, truncation = True, return_tensors = 'pt')
  inputs = {k: v.to(device) for k, v in inputs.items()}

  with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

  outputs = [{'scores' : value, 'index' : index} for value, index in zip(probabilities.tolist(), candidate_qa_index_list)]

  # 'scores'의 두 번째 값(연속적인 문장일 확률)에 따라 내림차순으로 정렬
  sorted_output = sorted(outputs, key = lambda x : x['scores'][1], reverse=True)

  output = sorted_output[0] #텍스트가 소속될 질문

  if output['scores'][1] > standard:
    #모순이 발생하지 않기 위해서 질문 속 질문으로 판별된 경우 즉시 qa_pair_dictionary에서 해당 질문을 삭제해야 함.
    if in_question_texts == True:
        del qa_pair_dictionary[index]

    qa_pair_dictionary[output['index']]['answer'].append(item['text'])
    qa_pair_dictionary[output['index']]['answer_withurl'].append(df_withurl[df_withurl['number'] == index]['text'].iloc[0])
    qa_pair_dictionary[output['index']]['respondent'].append(item['name'])

  # print('\n',qa_pair_dictionary)
  # print('\n',used_question_index)

print(qa_pair_dictionary)

data_result= [item for item in qa_pair_dictionary.values()]

#결과 데이터를 데이터프레임으로 변환
df_result = pd.DataFrame(data = data_result, columns = ['question', 'answer', 'question_withurl', 'answer_withurl', 'questioner', 'respondent', 'date'])
print(df_result)

#answer가 비어있는 행은 제거
df_result = df_result[df_result['answer'].apply(lambda x: x != [])]
print(len(df_result))

print(df_result.iloc[0]['answer'] == [])

print(df_result)

#csv 파일로 google drive에 저장
#long 파일의 경우 illegalcharactererror로 인해서 .csv 파일로 저장하기
save_path = '/content/drive/My Drive/judge_answer_result_KcBERT_short_questiontrained.csv'

df_result.to_csv(save_path)
