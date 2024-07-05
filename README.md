# [Dacon 웹 기사 추천 대회](https://dacon.io/competitions/official/236290)

## 최종 결과
- Public - 11th
- Private - 13th

웹 기사 추천에 협업 필터링을 베이스로 추천 목록을 생성하였습니다. 국가에 대한 정보를 조금 덜어냈을때 가장 좋은 결과를 얻을 수 있었습니다.

결과적으로 텍스트 임베딩을 활용한 방법은 좋은 방법이었으나 대회 기간 중 저는 임베딩 성능에 중점을 두다보니 임베딩 Size가 너무 큰 모델을 선정하여 본문에 대한 임베딩을 진행하지 못하여 이를 활용하지 못한 부분이 아쉬웠습니다. 실제로 비교적 높은 스코어를 기록한 로직에서 동일한 방법을 사용했으나 보다 작은 모델을 활용하여 진행하였고 이를 통해 보다 정교한 추천 모델을 만든 것을 확인할 수 있었습니다.

---

## 웹 기사 추천 핵심 알고리즘
- 유저 행동패턴 기반 협업 필터링
  - 단순히 기사를 조회한 것이 아닌 중복 조회 기록 반영
  - 중복 방문에 대한 log-scale-weight
    -> Score UP!!
- 접속 기록에 있는 유저의 국가 정보 반영
  -> Score Up!!
- 직접 작성한 글 제외
  -> 이번 대회에서는 Score Down... But 추천 목표에 따라 Up이 됨.
- 컨텐츠 기반 협업 필터링 (텍스트 임베딩을 통한 추천)
  -> Score Up!!

  
#### 협업 필터링 - Base
- 단순 방문 Matrix
  1. 단순히 유저가 해당 기사를 봤는지 여부를 0, 1 이진값으로 설정하여 Matrix 생성.
  2. 방문 기록을 바탕으로 유저간 유사성 Matrix 생성.
  3. 두 Matrix의 내적을 통해 추천 리스트 생성
~~~python
# 사용자-기사 행렬 생성
user_article_matrix = view_log.groupby(['userID', 'articleID']).size().unstack(fill_value=0)

# 사용자 간의 유사성 계산
user_similarity = cosine_similarity(user_article_matrix)

# 추천 리스트 점수 계산
user_predicted_scores = user_similarity.dot(user_article_matrix)
~~~

- 누적 방문 Matrix(Log Scale) -> Score Up!
  - 방문 기록을 보면 하나의 웹 기사에 최대 30번 이상의 방문 기록을 갖고 있는 유저들이 있음.
  - 이는 유저가 관심이 있는 기사에 대한 의미있는 정보로서 추천 기사를 만들어내는데 중요 정보임.
  - 하지만 이를 그대로 카운팅하여 반영한다면 유사성과 점수 계산 과정에서 숫자 차이가 너무 커져 정확한 추천 목록을 만들기 어려움. 이를 해결하기 위해 로그를 적용해 편향을 줄임.
~~~python
user_article_matrix = train_data_random.groupby(['userID', 'articleID']).size().reset_index(name='visit_count')
user_article_matrix['log_weight'] = user_article_matrix_train_random['visit_count'].apply(lambda x: np.log1p(x))
user_article_matrix = user_article_matrix_train_random.pivot(index='userID', columns='articleID', values='log_weight').fillna(0)
~~~

---


#### 국가 정보 반영 -> Score Up!
- 방문 기록에는 국가(Coutry)와 지역(Region)에 대한 정보가 있다. 그렇다면 이에 대한 정보를 활용해볼 수 있을까?
- 방문한 국가에 대한 가중치 값을 설정하여 Matrix에 반영해보자.
- 결론은 협업 필터링 Matrix에 이미 국가적 정보가 잠재적으로 반영이 되어 있었다. 국가에 대한 정보를 더할 경우 Score Down! 국가에 대한 정보를 뺄 경우 Score Up!
- 이를 통해 알 수 있었던 것은 유저들의 로그 기록에는 행동 패턴안에 이미 국가적인 편향 강하게 반영된 데이터였기에 이를 미세하게 제거했을때 보다 정교한 추천이 가능했음을 알 수 있음.
~~~python
user_countries = view_log_train.groupby('userID')['userCountry'].unique().apply(set)

...

# 가중치 초기화 값은 전체 가중치들의 평균
mean_similarity = np.mean(user_similarity_train_random[user_similarity_train_random > 0])

additional_weight = mean_similarity * 0.8 # 가중치는 GridSearch를 통해 최적값 탐색
print('Additional_CONTRY_WEIGHT : ', additional_weight)
user_index = user_article_matrix_train_random.index
for i, user_i in enumerate(user_index):
    countries_i = user_countries[user_i]
    for j, user_j in enumerate(user_index[i+1:], start=i+1):  # 대칭성을 고려하여 j를 i+1부터 시작
        countries_j = user_countries[user_j]
        if countries_i & countries_j:  # 두 사용자가 하나 이상의 공통 국가를 가지고 있는 경우
            user_similarity_train_random[i, j] -= (additional_weight)
            user_similarity_train_random[j, i] -= (additional_weight)  # 유사성 행렬은 대칭이므로
~~~

---


#### 직접 작성글 제외 -> Score Down..
-> **이번 대회는 결국 자기가 쓴 글에 대한 추천도 좋은 추천으로 보는 평가였음. 이 로직은 추천을 어떻게 해줄 것인지 방향성에 따라 좋은 로직이 될 수 있음.**

- 방문 기록을 보면 본인이 작성한 게시글에 대한 방문 횟수가 많은 유저들이 존재함.
- 유저가 쓴글과 관심있는 타 포스팅에는 다를 것이라 가정하였음.
- 그래서 추천 과정에서 직접 작성했을 경우 방문기록에서 제거하도록 설정
- 완전히 제거했을 경우 추천 목록 점수가 급격하게 감소하였음. 그래도 이를 반영하고자 Matrix생성 이전에 개수를 제한하여 반영하려 했지만 그대로 사용하는게 가장 좋았음.
~~~python
for user_id in view_log['userID'].unique():
    user_writed_articles = article_info[article_info['userID'] == user_id]['articleID'].values
    user_view_log = view_log[view_log['userID'] == user_id]
    self_view_df = user_view_log[user_view_log['articleID'].isin(user_writed_articles)]
    self_ivew_list = self_view_df.index.to_list()
    remain_idx = self_view_df.groupby(['userID','articleID']).head(999).index.to_list() # .head() 부분을 통해 개수 제어
    drop_idx += [idx for idx in self_ivew_list if idx not in remain_idx]
view_log_with_lang = view_log_with_lang.drop(index=drop_idx)
~~~

---

#### 텍스트 임베딩 - Score Down...
- 방문 기록에는 숫자료 표현한 정형 데이터들도 존재하지만 그렇지 못한 비정형 데이터(기사 제목, 내용)들이 존재함. 이를 반영할 수 없을까?
- 공개되어 있는 임베딩 모델을 사용하여 기사 제목간 유사도를 측정하여 이를 Matrix에 반영해보자!
- 임베딩 모델은 총 1024 size의 벡터 공간을 활용하는 bge-m3모델을 사용. 해당 모델은 OpenAI, Cohere에서 제공하는 유료 임베딩 모델에서 크게 뒤쳐지지 않는 성능으 보여주는 모델임.
- 임베딩 size가 너무 커서 본문 텍스트에 대한 임베딩 진행시 OOM 에러가 발생으로 본문 데이터에는 적용하지 않고 제목 데이터들로만 진행.
- 결론적으로 제목 텍스트만으로 계산한 임베딩 값들로는 추천목록을 만들기는 부족했다. 하지만 동일한 로직을 사용한 타 참가자의 경우 이 로직을 size가 작은 임베딩 모델을 활용해 본문까지 활용하였고 이를 통해 좀 더 좋은 결과를 얻었음.
~~~python
# Embedding 모델 활용을 위해 sentence-transformers 설치, Vectorstore를 쉽게 핸들링하기 위해 langchain, langchain_community 사용
# pip install sentence-transformers langchain langchain_community chroma


def embed_file(CACHE_DIR, model_name = 'BAAI/bge-m3'):
    
    
    model_kwargs = {
        "device": "cpu" # M1 이상 맥북은 'mps', GPU가 있다면 'gpu'로 설정
    }
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder = CACHE_DIR
    )
    
    return embeddings

embedding = embed_file('pretrained/')

docs = [
    Document(page_content = row['Title'], metadata = {'id' : row['articleID']}) for idx, row in article_info.iterrows()
]

vectorstore = Chroma.from_documents(docs, embedding) # 시간걸림

# 기사별 상위 5개 기사와 점수(유사도) 저장
reco_dics = {id : {'docs' : [], 'scores' : []} for id in view_log['articleID'].unique()}
for article_id in tqdm(view_log['articleID'].unique(), total = len(view_log['articleID'].unique())):
    title = article_info[article_info['articleID'] == article_id]['Title'].iloc[0]
    for relevance_data in vectorstore.similarity_search_with_relevance_scores(title, k =5):
        doc, score = relevance_data
        if  article_id != doc.metadata['id']:
            
            reco_dics[article_id]['docs'].append(doc.metadata['id'])
            reco_dics[article_id]['scores'].append(score)


# 유저가 봤던 기사들의 목록의 연관성 높은 기사와 점수를 가져와 합산후, 유저별 상위 5개 추천 항목 생성
person_reco_dics = {id : [] for id in view_log['userID'].unique()}
for user_id in tqdm(view_log['userID'].unique(), total = len(view_log['userID'].unique())):
    
    view_articles = view_log[view_log['userID'] == user_id]['articleID'].values
    
    reco_list_dict = defaultdict(int)
    for article_id in view_articles:
        for doc, score in zip(reco_dics[article_id]['docs'], reco_dics[article_id]['scores']):
            reco_list_dict[doc] += score

    # 상위 5개 목록만 각 유저 아이디에 추가
    person_reco_dics[user_id] = sorted(reco_list_dict, key=lambda x: x[1], reverse=True)[:5]
~~~













