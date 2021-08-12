# 기본 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# skleran 데이터셋에서 보스턴 주택 데이터셋 로딩
from sklearn import datasets
housing = datasets.load_boston()

# 딕셔너리 형태이므로, key 값을 확인
housing.keys()

# 판다스 데이터프레임으로 변환
data = pd.DataFrame(housing['data'], columns=housing['feature_names'])
target = pd.DataFrame(housing['target'], columns=['Target'])
# 데이터셋 크기
print(data.shape)
print(target.shape)