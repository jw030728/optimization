from sklearn.datasets import load_boston
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd



boston_data = load_boston() # 보스턴집값 데이터 불러오기

boston = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names) #dataframe운pandas에서 데이터를처리하기 위한 기준
boston['target'] = boston_data.target #target = 집값정보 y축값

train = boston.sample(frac=0.8, random_state=200) #frac=몇프로 데이터 샘플링 할건지
test = boston.drop(train.index)

scatter_matrix(boston.drop(columns=["ZN","INDUS","CHAS","NOX","RM","DIS","PTRATIO","RAD"]))
plt.show()







