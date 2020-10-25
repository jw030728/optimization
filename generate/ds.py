import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
import seaborn as sns

#데이터불러오기
df = pd.read_csv('generator.csv')
sns.lmplot('RPM','VIBRATION',data=df,hue='STATUS', fit_reg=False)
plt.show()

#학습/테스트 데이터분리
train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)


#로지스틱 회귀 분석 진행

logistic = LogisticRegression()
logistic.fit(train[['RPM','VIBRATION']], train['STATUS'])
score = logistic.score(test[['RPM','VIBRATION']], test['STATUS'])

print(score)


#테스트데이터사용

guess = pd.DataFrame(columns=['RPM','VIBRATION'])
guess.loc[0] = [900,100]
print(logistic.predict(guess))



