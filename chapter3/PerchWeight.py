import numpy as np

# 농어의 무게와 길이 데이터
# 넘파이 배열
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

# 데이터가 어떤 형태를 띠고 있는지 파악하기 위해 산점도 렌더링
import matplotlib.pyplot as plt

# 농어의 길이가 커짐에 따라 무게도 같이 늘어난다
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련 세트와 테스트 세트 데이터 생성
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42
)

# 훈련 세트는 2차원 배열이여야 한다 
# 파이썬에서는 1차원 배열의 크기는 원소가 1개인 튜플로 나타낸다 
# 예를 들어 [1, 2, 3]의 크기는 (3, )이다
# 이를 2차원 배열로 만들기 위해 억지로 하나의 열을 추가하면 배열의 크기가 (3, 1)이 된다
# 배열을 나타내는 방식만 달라졌을 뿐 배열에 있는 원소의 개수는 동일하게 3개다

# 넘파이 배열은 크기를 바꿀 수 있는 reshape() 메서드 제공
# 예제
test_array = np.array([1, 2, 3, 4])
print(test_array.shape)
# 1차원 배열 4개의 원소 보유
# (4, )

# (2, 2) 크기의 2차원 배열로 변환
# 바꾸려는 배열의 크기를 지정할 수 있다
# 원본 배열에 있는 원소의 개수와 다르면 에러가 발생한다
test_array = test_array.reshape(2, 2)
print(test_array.shape)

# 넘파이는 배열의 크기를 자동으로 지정하는 기능도 제공
# 크기에 -1을 지정하면 나머지 원소 개수로 모두 채우라는 의미 (원소의 갯수를 몰라도 해당 원소의 갯수만큼 배열을 만들어준다)
# 원소가 하나인 2차원 배열이 만들어진다 [[a], [b], [c]]
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)

# 사이킷런에서 k-최근접 이웃 회귀 알고리즘을 구현한 클래스는 KNeighborsRegressor
# 회귀 모델 훈련
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor()

# k-최근접 이웃 회귀 모델을 훈련
knr.fit(train_input, train_target)

# 테스트 세트의 점수 확인
print(knr.score(test_input, test_target))