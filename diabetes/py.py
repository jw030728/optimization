import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns


#데이터불러오기
df = pd.read_csv('diabetes.csv',names=['a','b','c','d','e','f','g','h','result']) #레이블 없을때 이름붙이기

sns.pairplot(data=df[['a','b','c','d','e','f','g','h','result']], hue='result')
plt.show()

#학습/테스트 데이터분리
train = df.sample(frac=0.8, random_state=200)  # frac값을 학습용 나머지 테스트
test = df.drop(train.index)


logistic = LogisticRegression(solver='newton-cg') # solver에 따라 확률 정확도 차이 보임 
logistic.fit(train[[ 'a','b','c','d','e','f','g','h'  ]], train['result'])
score = logistic.score(test[[ 'a','b','c','d','e','f','g','h'  ]], test['result'])

print(score)


# #테스트데이터사용

# guess = pd.DataFrame(columns=['perimeter_mean', 'perimeter_worst'])
# guess.loc[0] = [20, 30]
# print(logistic.predict(guess))


















