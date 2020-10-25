import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

#파일 불러오기
df = pd.read_csv('fruit_data_with_colors.csv')
label_count = len(df['fruit_name'].unique()) 

sns.lmplot('color_score','mass',data=df, hue='fruit_name', fit_reg=False)#lmplot은 한번에 두개 int만됨
plt.show()

#테스트 데이터 분리
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)

#n_neighbors 레이블
knn = KNeighborsClassifier(n_neighbors=label_count) #데이터 추가되면 -> 평균값변화 상대적인것이라 score 달라짐 
knn.fit(train[['color_score','mass']], train['fruit_name']) #학습
score = knn.score(test[['color_score','mass']], test['fruit_name']) #학습한걸 검증
print(score)


#레이블 예측
guess = pd.DataFrame(columns=['color_score','mass']) 
guess.loc[0] = [200,0.7]   # 이미 만들어진 레이블에 새로운 값을 넣어서 뭐가 나올지 해보는거

print(knn.predict(guess))



