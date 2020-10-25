import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

df = pd.read_csv('bank_marketing_full.csv', sep=';')
dfTrain = pd.read_csv('bank_marketing_simple.csv', sep=';')

df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day',
                                 'poutcome', 'month'])

dfTrain = pd.get_dummies(dfTrain, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day',
                                 'poutcome', 'month'])
print(df.columns.tolist())


train = dfTrain.sample(frac=0.8, random_state=200)  # frac값을 학습용 나머지 테스트
test = df.drop(train.index)

# onehot encoding이 목푯값= .... // 목푯값 train_y에 넣고 목푯값지운걸 x에 넣음
train_y = train['y']
del train['y']
train_x = train

test_y = test['y']
del test['y']
test_x = test
print(train_x)

logistic = LogisticRegression(solver='newton-cg')
logistic.fit(train_x, train_y)

score = logistic.score(test_x, test_y)

print(score)
