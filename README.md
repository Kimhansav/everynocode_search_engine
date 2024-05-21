# everynocode_search_engine
모두의노코드 검색엔진

2024년 3월 27일 ~ 2024년 5월 10일 : 1차 종료(기본적인 틀)

2024년 n월 ~ : 2차 진행 예정(멀티모달 추가 및 기존 기능 개선)

Figma Board : https://www.figma.com/board/j0LTHO16epMX4jWug51ZOV/BubbleProject?node-id=0%3A1&t=9EYnHiP8R8FaByyB-1


# 프로젝트 요약

## 작성한 코드(latest : 사용 중, old : 코드가 제대로 작동하지 않음, 사용하지 않음)
## Docker 이미지로 제작할 때 사용한 app.py, dockerfile, requirements.txt 이외에는 모두 구글 드라이브와 연결한 뒤 로컬에서 실행

BP_preprocess : latest, 카카오톡 대화내용과 커뮤니티 게시글을 전처리하는 코드

BP_spacing : old, 정확도 상승을 목표로 PyKoSpacing으로 띄어쓰기를 실행하는 코드

BP_make_dataset : latest, 모델 학습을 위해 데이터셋을 제작하는 코드

BP_train_models : latest, MLM, NSP, Text classification으로 모델들을 학습시키는 코드

BP_judge_question_bespin-global/klue-roberta-small-3i4k-intent-classification : old, 한글 기반인 3i4k 데이터셋으로 파인튜닝된 의도 분류 모델로 카카오톡 대화내용 중 질문을 선별하는 코드

BP_judge_question_KcBERT : latest, 파인튜닝한 모델로 카카오톡 텍스트 중 질문에 해당하는 텍스트를 선별하는 코드

BP_judge_answer_zeroshot_MoritzLaurer/mDeBERTa-v3-base-mnli-xnli_baseline : old, 영어 기반 모델로 zero-shot text classification을 수행하는 모델을 이용해 질문에 대한 답변을 선별하는 코드

BP_judge_answer_zeroshot_pongjin/roberta_with_kornli : old, 한글 기반 모델로 zero-shot text classification을 수행하는 모델을 이용해 질문에 대한 답변을 선별하는 코드

BP_judge_answer_KcBERT_nsp : latest, 파인튜닝한 모델로 카카오톡 비질문 텍스트를 질문 텍스트에 소속시키는 코드

BP_summary : latest, 카카오톡 대화내용 + 커뮤니티 질문-답변 쌍의 질문 요약본, 답변 요약을 생성하는 코드

BP_sentence_embed : latest, 질문-답변 쌍들의 임베딩 벡터를 생성하는 코드

BP_local_search_test : latest, 로컬에서 실행하는 유사도 검색 코드

BP_server_api : old, Flask 사용한 서버 코드

app.py : latest, Docker 이미지로 제작할 api 코드. Flask 사용함

dockerfile : latest, Docker 이미지를 제작할 때 필요한 요소들, GCP 서비스 계정의 키 파일 포함


# 훈련시킨 모델

Pretrained_Model : Target domain의 unlabeled corpus로 MLM 학습을 한 KcBERT, 데이터 47737개

Pretrained_Model_sentence_embed : Target domain의 unlabeled corpus로 MLM 학습을 한 KoSimCSE_BERT, 데이터 47737개, 실패

Finetuned_Model_judge_question : 질문 데이터셋으로 Sequence classification 학습을 한 Pretrained_Model, 데이터 3407개

Finetuned_Model_judge_answer : 질문-답변 데이터셋으로 NSP 학습을 한 Pretrained_Model, 데이터 5824개
