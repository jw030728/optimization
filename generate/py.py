import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('generator.csv')
sns.lmplot('RPM','VIBRATION',data=df,hue='STATUS', fit_reg=False)
plt.show()


train = df.sample(frac=0.8, random_state=200)
test=df.drop(train.index)


knn=KNeighborsClassifier(n_neighbors=2) #
knn.fit(train[['RPM','VIBRATION']], train['STATUS'])#RPM VIBRATION을통해 STATUS나옴 fit으로학습
score = knn.score(test[['RPM','VIBRATION']], test['STATUS']) #test로 얼마나 맞는지 1이면 train test비교한거 다맞는거

print(score)


guess = pd.DataFrame(columns=['RPM','VIBRATION'])
guess.loc[0] = [800,200]       #좌표보고 틀리면 faulty

print(knn.predict(guess))




