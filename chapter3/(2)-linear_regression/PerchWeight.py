import numpy as np

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 훈련 세트와 테스트 세트 데이터 생성 특성 데이터는 2차원 배열로 변환
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# 최근접 이웃 개수를 3으로 하는 모델을 훈련
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

# 무게 예측
print(knr.predict([[50]]))

# 예측값은 1033g으로 나오지만 실제 해당 농어의 무게는 1.5kg

# 이상 데이터 분석을 위한 훈련 세트와 50cm 농어, 이 농어의 최근접 이웃을 산점도에 표시
import matplotlib.pyplot as plt

# 50cm 농어의 이웃을 구한다
distances, indexes = knr.kneighbors([[50]])

# 훈련 세트와 선점도를 그린다
plt.scatter(train_input, train_target)

# 이웃 샘플을 그린다
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

# 50cm 농어
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 산점도를 그리면 50cm는 범위를 벗어나 있다

# 이웃 샘플의 타깃의 평균
print(np.mean(train_target[indexes]))

# 새로운 샘플이 훈련 세트의 범위를 벗어나면 엉뚱한 값을 예측할 수 있다
# 예를 들어 50cm가 아닌 100cm의 농어도 여전히 1033g으로 예측한다

print(knr.predict([[100]]))

# 100cm 농어의 산점도
distances, indexes = knr.kneighbors([[100]])

plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(100, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# k-최근접 이웃 알고리즘의 한계로 이 문제를 해결하려면 가장 큰 농어가 포함되도록 훈련 세트를 다시 만들어야 한다
# 사실 머신러닝 모델은 한 번 만들고 끝나는 프로그램이 아니다
# 시간과 환경이 변화하면서 데이터도 바뀌기 떄문에 주기적으로 새로운 훈련 데이터로 모델을 다시 훈련해야 한다
# 예를 들어 배달 음식이 도착하는 시간을 예측하는 모델은 배달원이 바뀌거나 도로 환경이 변할 수 있기 때문에 새로운 데이터를 사용해 반복적으로 훈련해야한다