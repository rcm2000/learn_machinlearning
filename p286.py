import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./auto-mpg.csv', header=None)

df.columns = ['mpg','cylinders','displacement','horsepower','weight',
              'acceleration','model year','origin','name']

# horsepower 열의 자료형 변경 (문자열 ->숫자)
df['horsepower'].replace('?', np.nan, inplace=True)      # '?'을 np.nan으로 변경
df.dropna(subset=['horsepower'], axis=0, inplace=True)   # 누락데이터 행을 삭제
df['horsepower'] = df['horsepower'].astype('float')

ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]

X=ndf[['weight']]  #독립 변수 X
y=ndf['mpg']     #종속 변수 Y

# train data 와 test data로 구분(7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print('훈련 데이터: ', X_train.shape)
print('검증 데이터: ', X_test.shape)
print('\n')

lr = LinearRegression()

# train data를 가지고 모형 학습
lr.fit(X_train, y_train)
r_square = lr.score(X_test, y_test)
print(r_square)
print('\n')

print('기울기 a: ', lr.coef_)
print('\n')

# 회귀식의 y절편
print('y절편 b', lr.intercept_)
print('\n')

#print(X);
y_hat = lr.predict(X)
save_data = lr

result = lr.predict(pd.DataFrame({'weight':[2278.7]}))
print(result)
with open("myClass.pickle", "wb") as w:
    pickle.dump(save_data, w)



# plt.figure(figsize=(10, 5))
# ax1 = sns.kdeplot(y, label="y")
# ax2 = sns.kdeplot(y_hat, label="y_hat", ax=ax1)
# plt.legend()
# plt.show()
# print(ndf)