from sklearn.datasets import fetch_california_housing
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


california_data = fetch_california_housing()
california = pd.DataFrame(data=california_data.data, columns=california_data.feature_names)
california['target'] = california_data.target

train = california.sample(frac=0.8, random_state=200)
test = california.drop(train.index)

scatter_matrix(california.drop(columns=['AveRooms','AveBedrms','Population','AveOccup']))
plt.show()

# mlr = LinearRegression()
# mlr.fit(train[['Medlnc','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']]
#         ,train['target'])#mlr.fit이 회귀분석 자동해주는거 train[x값 y값
# print(mlr.intercept_)#여기서 intercept는 절편 y절편
# print(mlr.coef_)
#
# sum_difference=0

# for i, row in test.iterrows():   #estimate집값예측
#     estimate = row['Medlnc'] * mlr.coef_[0] + row['AveRooms'] * mlr.coef_[1]+row['AveBedrms'] * mlr.coef_[2]+ \
#                row['Population'] * mlr.coef_[3] +row['AveOccup'] * mlr.coef_[4]+row['Latitude'] * mlr.coef_[5]+ \
#                row['Longitude'] * mlr.coef_[6]
#                + mlr.intercept_
#     #print(estimate - row['target'])
#     sum_difference=sum_difference+ abs(row['target'] -estimate)

