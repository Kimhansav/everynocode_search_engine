# 질문-답변 선별 & 모두의노코드 검색 엔진

 - 2024년 3월 27일 ~ 2024년 5월 10일 : 1차 종료(기본적인 틀)

 - 2024년 n월 ~ : 2차 진행 예정(정량 지표를 통해 모델의 성능 개선)

Figma Board : https://www.figma.com/board/j0LTHO16epMX4jWug51ZOV/BubbleProject?node-id=0%3A1&t=9EYnHiP8R8FaByyB-1

모델 성능 실험 기록.pptx에서 사용한 모델들의 비교를 확인할 수 있습니다.

<br/><br/><br/>


## 개요

  ![image](https://github.com/Kimhansav/everynocode_search_engine/blob/7b09f8839114638e18c5495fcaf2bc9247517929/BubbleProject%20(3).png)


bubble.io에 대한 카카오톡 대화문 원본 데이터와 커뮤니티 게시글 데이터를 입력하면 전처리부터 질문-답변 선별, 텍스트 요약, sentence 임베딩 생성까지 수행합니다. 얻은 데이터셋을 Google Cloud Storage에 업로드하고, app.py를 docker 이미지로 제작한 후 Google Cloud Run에 업로드하면 검색 엔진 API를 만들 수 있습니다. 이 API의 엔드포인트에 검색 문장을 GET 요청으로 전송하면 Storage의 데이터셋에서 이와 관련된 질문-답변을 유사도가 높은 순서로 반환받습니다.

<br/><br/><br/>


## 코드 및 모델 설명

### 코드

#### latest(사용 중)

<br/>

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
  <br/>

   - Pretraining Dataset

    1. Kss를 활용해서 데이터프레임의 각 열에 대해 해당 열에 소속된 텍스트들을 문장 단위로 분리하는 함수를 제작했습니다.
    
    2. 카카오톡 대화문 데이터, 커뮤니티 질문답변 게시글 데이터, 커뮤니티 전체 게시글 데이터, 커뮤니티 전체 댓글 데이터를 함수로 처리한 뒤 결과들을 모두 결합했습니다.

   - Finetuning Dataset(질문 선별 작업)

    1. 데이터셋의 레이블별로 토큰화된 길이가 다르다면 모델의 학습에 데이터의 길이가 영향을 줄 여지가 있습니다. 이를 방지하기 위해 각 데이터를 문장의 형태를 깨지 않도록 n개의 덩어리로 분리하는 함수를 제작했습니다.
    
    2. 커뮤니티 데이터 중에서는 positive sample로 질문, negative sample로 답변, 빌더로그 글을 사용했습니다. bubble.io에 대한 내용이 많이 포함된 평서문 데이터로 빌더로그 글이 적당했습니다. 그 다음 스퀘어와 쇼케이스 글의 경우 bubble.io 내용이 별로 포함되지 않았지만 올바른 평서문을 얻을 수 있기에 차선책으로 보류해 두었고, 자유 주제 글은 질문도 섞여 있었기에 추가적인 선별에 수작업이 필요해 제외했습니다.
    
    3. 커뮤니티 질문 데이터는 그대로 두고, 답변은 2등분, 빌더로그 글은 6등분한 뒤 카카오톡 대화문 중 직접 질문을 선별한 데이터셋과 결합해 질문 선별 학습 데이터셋을 완성했습니다. 
    
   - Finetuning Dataset(답변 선별 작업)

    1. 제작한 데이터셋에는 Negative sample이 없기 때문에, Positive sample의 n배만큼 Negative sample을 생성하는 함수를 제작했습니다. 전체 데이터를 다루는 인덱스를 활용해서, Positive sample에 사용된 데이터의 인덱스를 제외한 나머지 인덱스들에서 랜덤 추출을 실행해 Negative sample을 제작합니다.
    
    2. 커뮤니티 데이터셋의 경우 질문답변 게시글을 사용했습니다. 답변의 경우 하나의 질문에 대한 여러 답변 모두가 하나의 데이터프레임 셀 안에 들어있었습니다. 이를 문장 단위가 아닌 답변의 단위로 분리한 뒤, 선별된 답변을 질문 뒤에 하나하나 붙여가는 방식으로 Positive sample을 제작했습니다. 예시는 다음과 같습니다.
     - 질문이 A, 이에 대한 답변이 B, C, D 라고 가정합니다. 이때 만들어지는 Positive sample은 (질문, 답변인지 판별된 텍스트)의 형식으로 나타내면 (A, B), ((A+B), C), ((A+B+C), D) 가 됩니다.  
     
    3. Negative sample의 경우 위에서 제작한 n배 샘플링 함수를 통해 생성합니다. 예시는 다음과 같습니다.
     - 위의 과정에서 A, B, C, D를 활용해 Positive sample을 제작했습니다. Negative sample에서 활용되는 데이터는 전체 데이터 중에서 A, B, C, D를 제외한 뒤 n개의 데이터가 무작위로 선택됩니다. 예를 들어 n = 2로 설정한 뒤 ((A+B), C)에 적용할 데이터로 X, Y가 선택되었다고 가정합니다. 이때 만들어지는 Negative sample은 ((A+B), X), ((A+B), Y)가 됩니다. 이러한 샘플링 과정은 전체 Positive sample의 개수만큼 반복됩니다.
     
    4. 카카오톡 대화문 중 질문과 이에 대한 답변쌍 데이터를 직접 선별해 커뮤니티 데이터의 Positive sample 형식으로 제작했습니다. 이후 커뮤니티 데이터와 마찬가지로 인덱스를 이용한 Negative sample 생성 과정을 거쳤습니다. 커뮤니티 데이터셋과 카카오톡 데이터셋을 결합해 답변 선별 학습 데이터셋을 완성했습니다.
    
    
</details>

<details>
  <summary>BP_train_models</summary>
  <br/>
  MLM, NSP, Text classification으로 모델들을 학습시키고 성능을 평가하는 코드입니다. 학습시킨 모델은 beomi/kcbert-base와 BM-K/KoSimCSE-bert-multitask입니다.<br/> 
  <br/>
  기존 모델이 학습한 데이터와 타겟 도메인의 데이터가 사용하는 어휘가 크게 다르다고 판단했습니다. 이를 해결하기 위해 soynlp를 활용해 타겟 도메인의 Pretraining Dataset에서 도메인 특화 어휘를 추출했고, 이를 토크나이저의 사전에 추가했습니다. 이후 MLM을 통해 도메인에 적응시켰습니다.<br/>
  <br/>
  MLM 학습의 경우 학습 데이터 : 검증 데이터를 9 : 1로 설정했습니다. Sequence Classification과 NSP의 경우 학습 데이터 : 검증 데이터 : 테스트 데이터를 8 : 1 : 1로 설정했습니다. 이때 레이블 간 데이터 수의 불균형이 존재해 stratify 옵션을 사용했습니다.<br/>
  <br/>
  성능 평가의 경우 먼저 질문 선별 모델, 답변 선별 모델에 대해서 수행했습니다. 평가 기준은 Accuracy, Precision, Recall, F1 score, 학습 시 초당 처리한 스텝 수, 학습 시 초당 처리한 샘플 수, 테스트 시 초당 처리한 스텝 수, 테스트 시 초당 처리한 샘플 수입니다. 학습과 테스트에 사용된 GPU는 Colab의 T4입니다.<br/>
  <br/>
  모델에 대해 자세한 설명은 아래의 '훈련시킨 모델'에 있습니다.
  <br/>

</details>

<details>
  <summary>BP_judge_question_KcBERT</summary>
  <br/>
  파인튜닝한 모델로 카카오톡 텍스트 중 질문에 해당하는 텍스트를 선별하는 코드입니다.<br/>
  <br/>
 
  파이프라인 선언 시 설정된 하이퍼파라미터는 다음과 같습니다.
  ```python
  text_classifier = TextClassificationPipeline(
    tokenizer=finetuned_tokenizer,
    model=finetuned_model,
    top_k = 1,
    truncation = True,
    batch_size = 128,
    device = device
  )
  ```
  이후 결과 데이터셋에서 너무 짧은 질문(토큰화된 길이가 7 이하인 문장)은 답변으로 변경하는 과정을 거쳤습니다.
 
</details>

<details>
  <summary>BP_judge_answer_KcBERT_nsp</summary>
  <br/>
  파인튜닝한 모델로 카카오톡 대화문에서 질문이 아닌 텍스트를 질문 텍스트에 소속시키는 코드입니다.<br/>
  <br/>
  기존 알고리즘은 하나의 질문에 대해 질문 직후 n개(보통 20~30)의 텍스트에 대해서 각 텍스트가 답변인지 아닌지 판단했습니다. 하지만 이 경우 하나의 텍스트가 서로 다른 질문에 대한 답변으로 선별될 가능성이 있었습니다. 이때 답변을 하나의 질문에 종속시키고 나머지 질문에 대해서는 해당 답변을 제거한다면 제거된 답변 이후의 답변까지 영향을 받는 등 치명적인 문제가 발생했습니다.<br/>
  <br/>
  이를 방지하기 위해 질문에 대해서 답변인지 선별하는 방식이 아닌, 답변에 대해서 가장 어울리는 질문이 무엇인지 고르도록 알고리즘을 설계했습니다. 이 알고리즘은 만약 답변으로 판별된 텍스트가 질문일 경우 질문 리스트에서 이를 삭제하는 과정까지 수행합니다.<br/>
  <br/>
  파이프라인을 사용하지 않고 직접 알고리즘을 제작했습니다.

</details>

<details>
  <summary>BP_summary</summary>
  <br/>
  카카오톡 대화내용 + 커뮤니티 질문-답변 쌍의 질문 요약본, 답변 요약을 생성하는 코드입니다.<br/>
  <br/>
  모델은 EbanLee/kobart-summary-v3를 사용했습니다. 한글 요약을 수행하는 모델 중 이 모델이 말의 뉘앙스를 살리며 생성한 결과가 이상적인 목표와 가장 비슷했습니다.<br/>
  <br/>
 
  질문에 대한 요약을 생성할 때 설정한 하이퍼파라미터는 다음과 같습니다.
  ```python
  question_summary_ids = model.generate(
    input_ids = input_ids,
    attention_mask = attention_mask,
    bos_token_id = model.config.bos_token_id,
    eos_token_id = model.config.eos_token_id,
    length_penalty = 1.0,
    max_length = 100,
    min_length = 5,
    num_beams = 6,
    repetition_penalty = 1.5,
    no_repeat_ngram_size = 3,
  )
  ```

  답변에 대한 요약을 생성할 때 설정한 하이퍼파라미터는 다음과 같습니다.
  ```python
  answer_summary_ids = model.generate(
    input_ids = input_ids,
    attention_mask = attention_mask,
    bos_token_id = model.config.bos_token_id,
    eos_token_id = model.config.eos_token_id,
    length_penalty = 1.0,
    max_length = 200,
    min_length = 5,
    num_beams = 6,
    repetition_penalty = 1.5,
    no_repeat_ngram_size = 3,
  )
  ```

  각 과정에서 사용된 하이퍼파라미터는 추후 성능 개선 작업에서 수정될 예정입니다.
  
</details>

<details>
  <summary>BP_sentence_embed</summary>
  <br/>
  질문-답변 쌍들의 임베딩 벡터를 생성하는 코드입니다.<br/>
  <br/>
  SBERT의 Mean pooling과 비슷하게, 사용자의 입력에 대해 데이터의 질문뿐만 아니라 답변까지 유사도 계산 과정에서 고려하기 위해 질문의 임베딩과 답변의 임베딩을 가중합했습니다. 현재 가중합 방식은 (질문 * 0.8 + 답변 * 0.2)이며, 추후 수정할 계획입니다.

</details>

<details>
  <summary>BP_local_search_test</summary>
  <br/>
  로컬에서 실행하는 유사도 검색 코드입니다.<br/>

</details>

<br/>

#### old(사용X)

<details>
  <summary>BP_spacing</summary>
  <br/>
  정확도 상승을 목표로 PyKoSpacing으로 띄어쓰기를 실행하는 코드입니다.<br/>

</details>

<details>
  <summary>BP_judge_question_bespin-global/klue-roberta-small-3i4k-intent-classification</summary>
  <br/>
  한글 기반인 3i4k 데이터셋으로 파인튜닝된 의도 분류 모델로 카카오톡 대화내용 중 질문을 선별하는 코드입니다.<br/>

</details>

<details>
  <summary>BP_judge_answer_zeroshot_MoritzLaurer/mDeBERTa-v3-base-mnli-xnli_baseline</summary>
  <br/>
  영어 기반 모델로 zero-shot text classification을 수행하는 모델을 이용해 질문에 대한 답변을 선별하는 코드입니다.<br/>

</details>

<details>
  <summary>BP_judge_answer_zeroshot_pongjin/roberta_with_kornli</summary>
  <br/>
  한글 기반 모델로 zero-shot text classification을 수행하는 모델을 이용해 질문에 대한 답변을 선별하는 코드입니다.<br/>

</details>

<details>
  <summary>BP_server_api</summary>
  <br/>
  Flask 사용한 서버 코드입니다.<br/>

</details>

<br/><br/><br/>


### 훈련시킨 모델

 <details>
  <summary>Pretrained_Model</summary>
  <br/>
  Target domain의 unlabeled corpus로 MLM 학습을 한 KcBERT, 데이터 47737개
  <br/>
  타겟 도메인에 적응한 후 토큰 임베딩의 크기는 (44857, 768)입니다.<br/>
  <br/>
  
  학습 시 설정된 하이퍼파라미터는 다음과 같습니다.
  ```python
  training_args = TrainingArguments(
    output_dir = './results',
    evaluation_strategy = 'steps',
    eval_steps = 500,
    save_strategy = "steps",
    save_steps = 500,
    num_train_epochs = 3,
    save_total_limit = 3,
    per_device_eval_batch_size = 8,
    per_device_train_batch_size = 8,
    warmup_steps = 300, 
    weight_decay = 0.01, 
    logging_dir = "./logs",
    load_best_model_at_end = True
   )
   
   trainer = Trainer(
    model = KcBERT_model,
    args = training_args,
    train_dataset = pretrain_dataset['train'],
    eval_dataset = pretrain_dataset['test'],
    callbacks = [EarlyStoppingCallback(patience = 5)]
  )
  ```

 </details>

 <details>
  <summary>Pretrained_Model_sentence_embed</summary>
  <br/>
  Target domain의 unlabeled corpus로 MLM 학습을 한 KoSimCSE_BERT, 데이터 47737개, 실패
  <br/>
  타겟 도메인에 적응한 후 토큰 임베딩의 크기는 (43041, 768)입니다.<br/>
  <br/>

  학습 시 설정된 하이퍼파라미터는 다음과 같습니다.
  ```python
  training_args = TrainingArguments(
    output_dir = './results',
    evaluation_strategy = 'steps',
    eval_steps = 500,
    save_strategy = "steps",
    save_steps = 500,
    num_train_epochs = 3,
    save_total_limit = 3,
    per_device_eval_batch_size = 8,
    per_device_train_batch_size = 8,
    warmup_steps = 300, 
    weight_decay = 0.01, 
    logging_dir = "./logs",
    load_best_model_at_end = True
  )

  trainer = Trainer(
    model = KoSim_model,
    args = training_args,
    train_dataset = pretrain_dataset['train'],
    eval_dataset = pretrain_dataset['test'],
    callbacks = [EarlyStoppingCallback(patience = 5)]
  )
  ```

 </details>

 <details>
  <summary>Finetuned_Model_judge_question</summary>
  <br/>
  질문 데이터셋으로 Sequence classification 학습을 한 Pretrained_Model, 데이터 3407개<br/>
  <br/>

  StratifiedKFold를 적용하지 않고 학습할 때 설정된 하이퍼파라미터는 다음과 같습니다.
  ```python
  training_args = TrainingArguments(
    output_dir = './results',
    learning_rate = 5e-5,
    evaluation_strategy = 'steps',
    eval_steps = 100,
    save_strategy = "steps",
    save_steps = 100,
    num_train_epochs = 3,
    save_total_limit = 3,
    per_device_eval_batch_size = 8,
    per_device_train_batch_size = 8,
    warmup_steps = 100, 
    weight_decay = 0.01, 
    logging_dir = "./logs",
    load_best_model_at_end = True
  )

  trainer = Trainer(
    model = pretrained_model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    callbacks = [EarlyStoppingCallback(patience = 3)]
  )
  ```

  StratifiedKFold를 적용하고 학습할 때 설정된 하이퍼파라미터는 다음과 같습니다.
  ```python
  #n_splits를 통해 fold 개수 조정
  n_splits = 5

  # StratifiedKFold 설정
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
  for fold, (train_idx, val_idx) in enumerate(skf.split(question_train_dataset, question_train_dataset['label'])):

    # 훈련 세트와 검증 세트 분리
    question_train_dataset_fold = question_train_dataset.select(train_idx)
    question_val_dataset_fold = question_train_dataset.select(val_idx)

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy = "steps",
        eval_steps = 100,
        save_strategy = "steps",
        save_steps = 100,
        learning_rate = 5e-5,
        num_train_epochs = 3, 
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        weight_decay = 0.01, 
        logging_dir = "./logs",
        load_best_model_at_end = True

    )

    # 트레이너 초기화 및 훈련
    trainer = Trainer(
        model = pretrained_model,
        args = training_args,
        train_dataset = question_train_dataset_fold,
        eval_dataset = question_val_dataset_fold,
        callbacks = [EarlyStoppingCallback(patience = 3)]
    )

    trainer.train()
  ```

 </details>

 <details>
  <summary>Finetuned_Model_judge_answer</summary>
  <br/>
  질문-답변 데이터셋으로 NSP 학습을 한 Pretrained_Model, 데이터 5824개<br/>
  <br/>

  StratifiedKFold를 적용하지 않고 학습할 때 설정된 하이퍼파라미터는 다음과 같습니다.
  ```python
  training_args = TrainingArguments(
    output_dir = './results',
    learning_rate = 2e-5, 
    evaluation_strategy = 'steps',
    eval_steps = 200,
    save_strategy = "steps",
    save_steps = 200,
    num_train_epochs = 3,
    save_total_limit = 3,
    per_device_eval_batch_size = 8,
    per_device_train_batch_size = 8,
    warmup_steps = 200, 
    weight_decay = 0.01, 
    logging_dir = "./logs",
    load_best_model_at_end = True
  )

  trainer = Trainer(
    model = pretrained_model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    callbacks = [EarlyStoppingCallback(patience = 3)]
  )
  ```

  StratifiedKFold를 적용하고 학습할 때 설정된 하이퍼파라미터는 다음과 같습니다.
  ```python
  #n_splits를 통해 fold 개수 조정
  n_splits = 5

  # StratifiedKFold 설정
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
  for fold, (train_idx, val_idx) in enumerate(skf.split(answer_train_dataset, answer_train_dataset['label'])):

    # 훈련 세트와 검증 세트 분리
    answer_train_dataset_fold = answer_train_dataset.select(train_idx)
    answer_val_dataset_fold = answer_train_dataset.select(val_idx)
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy = "steps",
        eval_steps = 100,
        save_strategy = "steps",
        save_steps = 100,
        learning_rate = 5e-5,
        num_train_epochs = 3, 
        per_device_train_batch_size = 8,
        per_device_eval_batch_size = 8,
        weight_decay = 0.01, 
        logging_dir = "./logs",
        load_best_model_at_end = True

    )

    # 트레이너 초기화 및 훈련
    trainer = Trainer(
        model = pretrained_model,
        args = training_args,
        train_dataset = answer_train_dataset_fold,
        eval_dataset = answer_val_dataset_fold,
        callbacks = [EarlyStoppingCallback(patience = 3)]
    )

    trainer.train()
 ```

 </details>

<br/><br/><br/>


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

<br/><br/><br/>


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

<br/><br/><br/>


### API 제작

API를 제작할 때 Google Cloud Storage의 데이터셋을 참조할 수 있도록 해당 프로젝트의 서비스 계정의 키 파일(.json)을 dockerfile에 추가해야 합니다.
![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/13ef9a71-1391-406e-98a3-bf29d66e75df)

![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/ed016edf-e1c2-430e-ab7e-02b1c31212cd)

![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/f7f9a58d-1652-4990-b73e-d99f04387d98)

![image](https://github.com/Kimhansav/everynocode_search_engine/assets/134425555/663fe13b-0d51-4d08-bac6-6db39857bbe8)

dockerfile에서 키 파일의 경로를 설정하는 부분을 다운로드한 .json 파일의 경로와 일치하도록 수정하고 Docker 이미지로 제작한 뒤, Google Cloud Run에 업로드합니다. 

<br/><br/><br/>


## 추후 개선사항

### Text Classification(질문 선별, 답변 선별)

F1 score를 활용할 계획입니다. 훈련시킨 모델로 실제 데이터를 선별해본 결과 False Negative, False Positive가 골고루 나타났습니다.
특히 답변 선별 태스크에서는 체감상 잘못된 답변의 개수(False Positive)는 많이 줄었지만, 
실제 답변이 존재함에도 답변 리스트가 빈 경우(False Negative)가 많이 발생했다고 느꼈습니다. 이에 따라 False Negative를 많이 줄이는 방향으로 개선시키면 F1 score가 빠르게 증가할 것이라고 생각합니다.

+벤치마크 점수가 평균적으로 더 높은 KcELECTRA도 테스트할 예정입니다.

<br/>

### Summarization(질문-답변 쌍 요약)

RDASS를 활용할 계획입니다. 요약 모델의 경우 ROUGE를 사용할 수도 있지만, 영어와 달리 한글의 경우 어근에 붙은 접사가 단어의 역할을 결정하며 단어의 변형이 자주 일어나므로 RDASS가 잘 작동할 확률이 높습니다.
이때 문제는 데이터셋 제작 비용이 높다는 것입니다. 이를 해결하기 위해 GPT와 같이 이미 어느 정도 타겟 도메인에 대해 학습한 모델이 요약을 진행하게 하고, 얻은 요약본을 요약 모델이 학습하는 방법 등을 사용하려고 합니다.

<br/>

### Feature Extraction(유사도 계산을 위한 문장 임베딩 생성)

FAISS를 통해서 유사도를 계산하기 때문에, 성능 지표로 FAISS가 활용하는 코사인 유사도, 유클리드 거리 등의 기법을 활용할 예정입니다.
이 방식도 마찬가지로 GPT와 같은 모델을 활용할 계획입니다. 유사도 점수를 0에서 5 사이로 나타내도록 데이터셋을 제작한다면 성능을 크게 높일 수 있겠지만 이는 쉽지 않다고 생각하기에, 
주어진 텍스트에 대해서 GPT가 구나 문장의 형태로 의미가 비슷하거나 다른 텍스트를 생성하게 한 뒤 각각 1과 0으로 라벨링하려고 합니다.
