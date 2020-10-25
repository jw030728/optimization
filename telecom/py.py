import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns


#데이터불러오기
df = pd.read_csv('telecom_churn.csv')
sns.lmplot('tenure', 'MonthlyCharges', data=df,
           hue='Churn')
plt.show()

#학습/테스트 데이터분리
train = df.sample(frac=0.8, random_state=200)  # frac값을 학습용 나머지 테스트
test = df.drop(train.index)

#로지스틱 회귀 분석 진행
logistic = LogisticRegression(solver='newton-cg')  # solver에 따라 확률 정확도 차이 보임
logistic.fit(train[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport',
  'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']], train['Churn'])
score = logistic.score(test[[ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges' ]], test['Churn'])

print(score)


# #테스트데이터사용

# guess = pd.DataFrame(columns=['perimeter_mean', 'perimeter_worst'])
# guess.loc[0] = [20, 30]
# print(logistic.predict(guess))







