# 질문-답변 선별 & 모두의노코드 검색 엔진

 - 2024년 3월 27일 ~ 2024년 5월 10일 : 1차 종료(기본적인 틀)

 - 2024년 n월 ~ : 2차 진행 예정(멀티모달 추가 및 기존 기능 개선)

Figma Board : https://www.figma.com/board/j0LTHO16epMX4jWug51ZOV/BubbleProject?node-id=0%3A1&t=9EYnHiP8R8FaByyB-1



## How to use
 
 - 구글 계정으로 로그인해 Google Cloud Platform에 새 프로젝트를 만듭니다. 이 프로젝트의 Cloud Run에는 Docker 이미지가, Cloud Storage에는 검색 엔진에 사용될 데이터셋이 업로드됩니다.


 - 카카오톡 데이터 : 카카오톡 방 -> 채팅방 설정 -> 대화 내용 내보내기 -> 모든 메시지 내부저장소에 저장 의 순서대로 실행한 후 원본 .txt 파일을 Google Drive에 업로드합니다.


 - 커뮤니티 데이터 : 코드의 전처리 규칙에 맞도록 커뮤니티 게시글 데이터를 .xlsx 및 .csv 파일로 추출한 뒤 Google Drive에 업로드합니다. 전처리 코드가 작동하는 게시글 데이터 형식은 다음과 같습니다.
  
  1. 커뮤니티 전체 게시글 데이터(.csv)
   - ![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/34da45fe-62cb-4644-843f-83b9692c35f2)
  
  2. 커뮤니티 전체 댓글 데이터(.csv)
   - ![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/dea07a32-0d58-4570-ae4b-049ebb72da22)

  3. 커뮤니티 질문답변 게시글 데이터(.xlsx)
   - ![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/ddad5e1a-0b2b-40e6-93c4-e6afcff5ae38)

 
 - latest의 코드들을 Google Drive의 '내 드라이브'에 저장합니다. 아래의 설명대로 BP_preprocess부터 순서대로 코드를 실행해 검색 엔진에 활용될 데이터셋을 얻습니다. 

#### 데이터셋 제작
  
#### API 제작

## 코드 및 모델 설명

### 코드

#### latest(사용중)

 - BP_preprocess : 카카오톡 대화내용과 커뮤니티 게시글을 전처리하는 코드

 - BP_make_dataset : 모델 학습을 위해 데이터셋을 제작하는 코드

 - BP_train_models : MLM, NSP, Text classification으로 모델들을 학습시키는 코드

 - BP_judge_question_KcBERT : 파인튜닝한 모델로 카카오톡 텍스트 중 질문에 해당하는 텍스트를 선별하는 코드

 - BP_judge_answer_KcBERT_nsp : 파인튜닝한 모델로 카카오톡 비질문 텍스트를 질문 텍스트에 소속시키는 코드

 - BP_summary : 카카오톡 대화내용 + 커뮤니티 질문-답변 쌍의 질문 요약본, 답변 요약을 생성하는 코드

 - BP_sentence_embed : 질문-답변 쌍들의 임베딩 벡터를 생성하는 코드

 - BP_local_search_test : 로컬에서 실행하는 유사도 검색 코드

#### old(사용X)

 - BP_spacing : 정확도 상승을 목표로 PyKoSpacing으로 띄어쓰기를 실행하는 코드

 - BP_judge_question_bespin-global/klue-roberta-small-3i4k-intent-classification : 한글 기반인 3i4k 데이터셋으로 파인튜닝된 의도 분류 모델로 카카오톡 대화내용 중 질문을 선별하는 코드

 - BP_judge_answer_zeroshot_MoritzLaurer/mDeBERTa-v3-base-mnli-xnli_baseline : 영어 기반 모델로 zero-shot text classification을 수행하는 모델을 이용해 질문에 대한 답변을 선별하는 코드

 - BP_judge_answer_zeroshot_pongjin/roberta_with_kornli : 한글 기반 모델로 zero-shot text classification을 수행하는 모델을 이용해 질문에 대한 답변을 선별하는 코드

 - BP_server_api : Flask 사용한 서버 코드


#### 훈련시킨 모델

 - Pretrained_Model : Target domain의 unlabeled corpus로 MLM 학습을 한 KcBERT, 데이터 47737개

 - Pretrained_Model_sentence_embed : Target domain의 unlabeled corpus로 MLM 학습을 한 KoSimCSE_BERT, 데이터 47737개, 실패

 - Finetuned_Model_judge_question : 질문 데이터셋으로 Sequence classification 학습을 한 Pretrained_Model, 데이터 3407개

 - Finetuned_Model_judge_answer : 질문-답변 데이터셋으로 NSP 학습을 한 Pretrained_Model, 데이터 5824개
