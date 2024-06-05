# 질문-답변 선별 & 모두의노코드 검색 엔진

 - 2024년 3월 27일 ~ 2024년 5월 10일 : 1차 종료(기본적인 틀)

 - 2024년 n월 ~ : 2차 진행 예정(멀티모달 추가 및 기존 기능 개선)

Figma Board : https://www.figma.com/board/j0LTHO16epMX4jWug51ZOV/BubbleProject?node-id=0%3A1&t=9EYnHiP8R8FaByyB-1

## 개요

bubble.io에 대한 카카오톡 대화문 원본 데이터와 커뮤니티 게시글 데이터를 입력하면 전처리부터 질문-답변 선별, 텍스트 요약, sentence 임베딩 생성까지 수행합니다. 얻은 데이터셋을 Google Cloud Storage에 업로드하고, app.py를 docker 이미지로 제작한 후 Google Cloud Run에 업로드하면 검색 엔진 API를 만들 수 있습니다. 이 API의 엔드포인트에 검색 문장을 GET 요청으로 전송하면 Storage의 데이터셋에서 이와 관련된 질문-답변을 유사도가 높은 순서로 반환받습니다.

## 코드 및 모델 설명

### 코드

#### latest(사용 중)

 <details>
  <summary>BP_preprocess</summary>
  <br/>
  카카오톡 대화내용과 커뮤니티 게시글을 전처리하는 코드입니다.<br/>
  <br/>
  
   - 카카오톡 대화 원본 데이터 처리 과정
     
    1. 원본 .txt 파일에서 아래 형식의 텍스트를 다음 결과의 형태로 전처리합니다.
       > '2023년 6월 15일 오후 2:27, Kimhansav : 안녕하세요, 신입 들어왔습니다!' ---> '2023년 6월 15일 오후 2:27', 'Kimhansav', '안녕하세요, 신입 들어왔습니다!'
       이후 날짜 문자열을 비교할 수 있도록 YYYY-MM-DD의 형식으로 변형합니다.
   
    2. 한 사람이 연속적으로 메시지를 보낸 경우 이들을 하나의 메시지로 통합합니다. 문맥 보존을 간편히 하기 위해서입니다.

    3. 메시지 내용의 경우 다음 규칙에 따라 전처리를 진행합니다.
     i. \U0001F600-\U0001F64F에 해당하는 유니코드 이모티콘을 제거합니다.
     ii. 사용자를 '@이름' 의 형태로 태그한 텍스트를 제거합니다.
     iii. 메시지에 '.png', '.jpg', '삭제된 메시지입니다', '사진 읽지 않음', '동영상 읽지 않음' 을 포함하면 이를 제거합니다. 혹은 '사진','사진 n장','동영상' 만이 존재하는 행의 경우 이를 제거합니다.
     iiii. 메시지의 첫 글자가 '['라면 해당 메시지 전체를 제거합니다. 제가 사용한 데이터에서 대부분의 광고 메시지가 이 형식을 따름을 확인했습니다.
     iiiii. 줄바꿈 문자 '\n'을 제거합니다.
  
   - 커뮤니티 게시글 데이터(질문답변 게시글, 전체 게시글, 전체 댓글) 처리 과정

    1. 질문답변 게시글 데이터에서 원본 질문글로 이동할 수 있게 하기 위해 Slug를 변형한 링크를 추가합니다.
   
    2. _x1008_와 같은 기호를 자동으로 제거하기 위해 전체 게시글 데이터와 전체 댓글 데이터를 cp949 형식으로 읽은 뒤 다시 utf-8 형식으로 읽습니다.

    3. 게시글 작성 일자를 카카오톡 텍스트 생성 일자와 비교할 수 있도록 YYYY-MM-DD의 형식으로 변형합니다.
   
    4. 글 내용의 경우 다음 규칙에 따라 전처리를 진행합니다.
     i. '[ul]', '[ol]'과 같은 태그가 많아 []에 둘러싸인 텍스트를 set()에 입력한 후 태그 종류를 조사합니다. [] 안에 중요한 정보가 들어있는 경우도 있기 때문에 직접 제거할 태그를 선별했습니다.
     ii. 줄바꿈 문자, url 형식, 이미지 형식 텍스트를 제거합니다.
     iii. \U0001F600-\U0001F64F에 해당하는 유니코드 이모티콘을 제거합니다.
</details>

<details>
  <summary>BP_make_dataset</summary>
  <br/>
  모델 학습을 위한 데이터셋을 제작하는 코드입니다. Pretraining을 위한 데이터셋, Finetuning을 위한 데이터셋이 있습니다.<br/>

  - 
</details>

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


### 훈련시킨 모델

 - Pretrained_Model : Target domain의 unlabeled corpus로 MLM 학습을 한 KcBERT, 데이터 47737개

 - Pretrained_Model_sentence_embed : Target domain의 unlabeled corpus로 MLM 학습을 한 KoSimCSE_BERT, 데이터 47737개, 실패

 - Finetuned_Model_judge_question : 질문 데이터셋으로 Sequence classification 학습을 한 Pretrained_Model, 데이터 3407개

 - Finetuned_Model_judge_answer : 질문-답변 데이터셋으로 NSP 학습을 한 Pretrained_Model, 데이터 5824개

## How to use
 
구글 계정으로 로그인해 Google Cloud Platform에 새 프로젝트를 만듭니다. 이 프로젝트의 Cloud Run에는 Docker 이미지가, Cloud Storage에는 검색 엔진에 사용될 데이터셋이 업로드됩니다.

카카오톡 데이터 : 카카오톡 방 -> 채팅방 설정 -> 대화 내용 내보내기 -> 모든 메시지 내부저장소에 저장 의 순서대로 실행한 후 원본 .txt 파일을 Google Drive에 업로드합니다.

커뮤니티 데이터 : 코드의 전처리 규칙에 맞도록 커뮤니티 게시글 데이터를 .xlsx 및 .csv 파일로 추출한 뒤 Google Drive에 업로드합니다. 전처리 코드가 작동하는 게시글 데이터 형식은 다음과 같습니다.
  
 1. 커뮤니티 전체 게시글 데이터(.csv)
  ![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/34da45fe-62cb-4644-843f-83b9692c35f2)
  
 2. 커뮤니티 전체 댓글 데이터(.csv)
  ![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/dea07a32-0d58-4570-ae4b-049ebb72da22)

 3. 커뮤니티 질문답변 게시글 데이터(.xlsx)
  ![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/ddad5e1a-0b2b-40e6-93c4-e6afcff5ae38)

latest의 코드들을 Google Drive의 '내 드라이브'에 저장합니다. 아래의 설명을 따라 BP_preprocess부터 순서대로 코드를 실행해 검색 엔진에 활용될 데이터셋을 얻습니다. 

### 데이터셋 제작

**주의사항** : 각 코드에서 결과 데이터를 생성해 다음 코드에서 이를 읽는 방식이므로, Google Drive에 업로드하는 결과물의 경로를 수정할 경우 현재 코드에서 저장하는 파일의 경로와 다음 코드에서 호출하는 파일의 경로가 일치해야 합니다.

다음의 순서대로 코드를 하나씩 실행합니다.

<details>
  <summary>1. BP_preprocess</summary>
 
  파일을 호출하는 코드 블록에서 Google Drive에 업로드한 카카오톡 원본과 커뮤니티 게시글 원본의 이름을 변수로 설정해야 합니다. 예시는 다음과 같습니다.

  ```python
  drive.mount('/content/drive')
  file_path = '/content/drive/My Drive/KakaoTalkChats-1.txt'
  ```
  ```python
  qna_path = '/content/drive/My Drive/community_qna.xlsx' #커뮤니티 질문답변 게시글 데이터
  all_contents_path = '/content/drive/My Drive/community_all_contents.csv' #커뮤니티 전체 게시글 데이터
  all_comments_path = '/content/drive/My Drive/community_all_comments.csv' #커뮤니티 전체 댓글 데이터
  ```
  
  결과물 저장 경로를 설정합니다. 예시는 다음과 같습니다.
  
  ```python
  #.xlsx 파일로 카카오톡 전처리 결과를 google drive에 저장
  talk_save_path = '/content/drive/My Drive/talk_preprocess_result_short.xlsx'
  df.to_excel(talk_save_path)
  ```
  ```python
  #전처리된 세 커뮤니티 데이터 파일을 google drive에 저장
  df_qna.to_excel('/content/drive/My Drive/community_qna_preprocessed.xlsx')
  df_all_contents.to_csv('/content/drive/My Drive/community_all_contents_preprocessed.csv')
  df_all_comments.to_csv('/content/drive/My Drive/community_all_comments_preprocessed.csv')
  ```
  
  이후 GPU를 사용할 필요 없이 CPU로 전체 코드를 실행합니다.

  Google Drive에 업로드된 커뮤니티 데이터 전처리 결과 파일 세 개를 다운로드합니다. 간혹 이미지 인코딩 텍스트가 너무 길어 여러 셀로 나누어진 경우가 존재합니다. 전처리 함수는 데이터프레임의 셀 단위로 작동하기에 이를 처리하지 못하며, 직접 제거해주어야 합니다.
  이미지 인코딩 텍스트를 제거한 뒤 세 파일 모두 .xlsx 형식으로 다시 Google Drive에 업로드합니다.
</details>

<details>
  <summary>2. BP_make_dataset</summary>

  Google Drive에 업로드한 전처리 결과 파일을 호출합니다. 
  직접 레이블링한 데이터셋을 업로드해야 합니다. 질문 데이터셋은 질문 선별 모델이, 질문답변 데이터셋은 답변 선별 모델이 학습할 예정입니다. 예시는 다음과 같습니다.
  ```python
  #직접 제작한 카카오톡 질문 데이터셋 로드
  talk_question_finetune_path = '/content/drive/My Drive/talk_finetune_question_dataset.xlsx'
  #직접 제작한 카카오톡 질문답변 데이터셋 로드
  talk_finetune_path = '/content/drive/My Drive/talk_finetune_dataset.xlsx'
  ```
  GPU를 사용할 필요 없이 CPU로 전체 코드를 실행합니다.
</details>

<details>
  <summary>3. BP_train_models</summary>
 
  Huggingface에서 모델을 불러와 직접 제작한 학습 데이터셋으로 학습시킵니다.
  런타임 유형을 GPU로 변경한 후 전체 코드를 실행합니다.
</details>

<details>
  <summary>4. BP_judge_question_KcBERT</summary>
 
  학습시킨 모델을 통해 카카오톡 데이터에서 텍스트 분류를 진행합니다.
  런타임 유형을 GPU로 변경한 후 전체 코드를 실행합니다.
</details>

<details>
  <summary>5. BP_judge_answer_KcBERT_nsp</summary>
 
  학습시킨 모델을 통해 카카오톡 데이터에서 텍스트 분류를 진행합니다.
  런타임 유형을 GPU로 변경한 후 전체 코드를 실행합니다.
</details>

<details>
  <summary>6. BP_summary</summary>
 
  질문-답변 선별 결과 데이터와 커뮤니티 질문답변 게시글 데이터를 결합한 뒤 각 질문-답변 쌍에 질문 요약, 답변 요약을 생성 후 추가합니다.
  런타임 유형을 GPU로 변경한 후 전체 코드를 실행합니다.
</details>

<details>
  <summary>7. BP_sentence_embed</summary>
 
  질문 요약, 답변 요약 생성 결과 데이터에서 각 질문-답변 쌍의 질문 원본과 답변 원본을 일정 비율로 반영해 임베딩을 생성합니다.
  런타임 유형을 GPU로 변경한 후 전체 코드를 실행합니다.
</details>


### API 제작

API를 제작할 때 Google Cloud Storage의 데이터셋을 참조할 수 있도록 해당 프로젝트의 서비스 계정의 키 파일(.json)을 dockerfile에 추가해야 합니다.
![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/13ef9a71-1391-406e-98a3-bf29d66e75df)

![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/ed016edf-e1c2-430e-ab7e-02b1c31212cd)

![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/f7f9a58d-1652-4990-b73e-d99f04387d98)

![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/663fe13b-0d51-4d08-bac6-6db39857bbe8)

dockerfile에서 키 파일의 경로를 설정하는 부분을 다운로드한 .json 파일의 경로와 일치하도록 수정하고 Docker 이미지로 제작한 뒤, Google Cloud Run에 업로드합니다. 
