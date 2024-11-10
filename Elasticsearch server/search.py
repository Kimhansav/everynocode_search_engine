import json
from pprint import pprint
import os
import time
import torch

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, shape_list, TFBertModel, RobertaTokenizerFast, RobertaForSequenceClassification, TextClassificationPipeline, pipeline, BertTokenizer, BertForNextSentencePrediction,  TrainingArguments, BertForMaskedLM, Trainer, TrainerCallback



load_dotenv()


class Search:
    def __init__(self):
        self.model = AutoModel.from_pretrained('BM-K/KoSimCSE-bert-multitask') #훈련시킨 임베딩 생성 모델을 불러오기
        self.tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-bert-multitask')

        # self.es = Elasticsearch('https://localhost:9200', 로컬의 경우 localhost:9200으로 통신
        #                         basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")), verify_certs=False) 

        #gcp 버전(외부ip)
        # self.es = Elasticsearch('http://34.47.80.218:9200', 
        #                         basic_auth=("elastic", 'votmdnjem'), verify_certs=False) 

        #gcp 수정버전(내부ip)
        self.es = Elasticsearch('http://34.47.80.218:9200', #gcp에 업로드할 경우 내부 ip로 통신
                                basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")), verify_certs=False) 


        #이렇게 설정하면 바로 gcp vm 인스턴스에 연결 가능!
        # self.es = Elasticsearch('http://34.47.80.218:9200', 
        #                         basic_auth=("elastic", os.getenv("ELASTIC_PASSWORD")))  
        
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def create_index(self):
        self.es.indices.delete(index='my_documents', ignore_unavailable=True)
        self.es.indices.create(index='my_documents', mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                    # 'index_options': {
                    #     'type': 'hnsw',
                    # }
                }
            }
        })

    #검색 함수에서 사용할 사용자 입력 인코딩 함수
  

    def get_embedding(self, text):
        # return self.model.encode(text) #모델 로드해서 그 모델로 text encode 진행
        try:
            inputs = self.tokenizer(text, padding = True, truncation = True, return_tensors = 'pt')

            with torch.no_grad():
                    outputs = self.model(**inputs)

            #배치 단위가 아니라 단일 문장이기 때문에 mean pooling을 할 때 dim = 0으로 설정
            mean_pooling = torch.mean(outputs.last_hidden_state, dim = 0)
            # print(mean_pooling)
            return mean_pooling[0].tolist()
        except Exception as e:
            # app.logger.error(f'Error in encode: {str(e)}')
            # raise e
            print(e)

    def insert_document(self, document):
        return self.es.index(index='my_documents', document={
            **document,
            # 'embedding': self.get_embedding(document['summary']), #튜토리얼 코드는 요약문으로 임베딩 생성?
        })
    
    def insert_documents(self, documents):
        operations = []
        for document in documents:
            operations.append({'index': {'_index': 'my_documents'}})
            operations.append({
                **document,
                # 'embedding': self.get_embedding(document['summary']), #튜토리얼 코드는 요약문으로 임베딩 생성?
            })
        return self.es.bulk(operations=operations)
    
    def reindex(self):
        self.create_index()
        with open('sentence_embed_result_short.json', 'rt', encoding = 'utf-8') as f: #인덱싱할 파일 선택 기존에는 data.json
            documents = json.loads(f.read())
        return self.insert_documents(documents)
    
    def search(self, **query_args):
        print(self.es.search(index='my_documents', **query_args))
        return self.es.search(index='my_documents', **query_args)
    
    def retrieve_document(self, id):
        return self.es.get(index='my_documents', id=id)
    