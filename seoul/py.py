import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('seoul.csv')
label_count = len(df['name'].unique())   #name 값중에서 처음나오는 것만뽑음 unique가

sns.lmplot('lat','lon',data=df, hue='name', fit_reg=False)
plt.show()

#테스트 데이터 분리
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)


#n_neighbors 레이블
knn = KNeighborsClassifier(n_neighbors=label_count) #데이터 추가되서 평균값변화 score 달라짐
knn.fit(train[['lat','lon']], train['name'])
score = knn.score(test[['lat','lon']], test['name'])
print(score)


#레이블예측
guess =  pd.DataFrame(columns=['lat','lon'])
guess.loc[0] = [37.520040, 127.110136]
print(knn.predict(guess))
























