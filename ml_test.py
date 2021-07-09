from itertools import count
from numpy.lib.function_base import iterable
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df2 = pd.read_csv('./model/tmdb.csv', encoding='utf-8')

count = TfidfVectorizer(stop_words = "english")
# count = CountVectorizer(stop_words = "english")
count_matrix = count.fit_transform(df2['soup'])
cos_sim = cosine_similarity(count_matrix, count_matrix)

indices = pd.Series(df2.index, index= df2['title'])

idx = indices['Avatar']
sim_scores = list(enumerate(cos_sim[idx]))
sim_scores = sorted(sim_scores, key= lambda x:x[1], reverse=True) ## cosine 계수(유사도)를 내림차순으로 정렬
sim_scores = sim_scores[1:11] ## 유사도 top10만 뽑음
sim_indexs = [i[0] for i in sim_scores] ## 유사도의 인덱스 값을 뽑아냄
tit = df2['title'].iloc[sim_indexs]
dat = df2['release_date'].iloc[sim_indexs]
## 유사도 top10의 제목과 개봉연도만 뽑아냄
return_df = pd.DataFrame(columns=['Title', 'Year']) ## 빈 배열 만들기 
return_df['Title'] = tit
return_df['Year'] = dat
## 빈배열에 채워넣기

print(return_df)

