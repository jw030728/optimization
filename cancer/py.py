import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns


#데이터불러오기
df = pd.read_csv('breast_cancer.csv')
sns.lmplot('radius_mean', 'texture_mean', data=df,
           hue='diagnosis', fit_reg=False)
plt.show()


#학습/테스트 데이터분리
train = df.sample(frac=0.8, random_state=200)  # frac값을 학습용 나머지 테스트
test = df.drop(train.index)


#로지스틱 회귀 분석 진행
logistic = LogisticRegression(solver='newton-cg') # solver에 따라 확률 정확도 차이 보임 
logistic.fit(train[[  "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se",
                    "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", ]], train['diagnosis'])
score = logistic.score(test[[  "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
                             "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst", ]], test['diagnosis'])

print(score)


# #테스트데이터사용

# guess = pd.DataFrame(columns=['perimeter_mean', 'perimeter_worst'])
# guess.loc[0] = [20, 30]
# print(logistic.predict(guess))
