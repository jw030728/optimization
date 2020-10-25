import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns


df = pd.read_csv('sharing_bike_train.csv')

# sns.pairplot(data=df[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered']], hue='count')
# plt.show()

df['year'] = pd.to_datetime(df['datetime']).dt.year
df['month'] = pd.to_datetime(df['datetime']).dt.month
df['day'] = pd.to_datetime(df['datetime']).dt.day
df['hour'] = pd.to_datetime(df['datetime']).dt.hour

train = df.sample(frac=0.8, random_state=200)
test = df.drop(train.index)


mlr = LinearRegression()
mlr.fit(train[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered']]
        ,train['count'])#mlr.fit이 회귀분석 자동해주는거 train[x값 y값
print(mlr.intercept_)#여기서 intercept는 절편 y절편
print(mlr.coef_)

sum_difference=0

for i, row in test.iterrows():   
    estimate = row['season'] * mlr.coef_[0] + row['holiday'] * mlr.coef_[1]+row['workingday'] * mlr.coef_[2]+ \
               row['weather'] * mlr.coef_[3] +row['temp'] * mlr.coef_[4]+row['atemp'] * mlr.coef_[5]+row['humidity'] * mlr.coef_[6]+ \
               row['windspeed'] * mlr.coef_[7] + row['casual'] * mlr.coef_[8]+row['registered'] * mlr.coef_[9]+ mlr.intercept_
    #print(estimate - row['count'])
    sum_difference=sum_difference+ abs(row['count'] -estimate)
print(sum_difference)








