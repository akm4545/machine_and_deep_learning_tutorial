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

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 최적의 직선보다 최적의 곡선을 찾아야 한다
# 2차 방정식의 그래프를 그리려면 길이를 제곱한 항이 훈련 세트에 추가되어야 한다
# 무게 = a * 길이² + b * 길이 + c
# 농어의 길이를 제곱해서 원래 데이터 앞에 붙인다

# 넘파이의 column_stack() 함수 사용
# train_input을 제곱한 것과 train_input 두 배열을 나란히 붙인다 
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

# train_input ** 2 식에도 넘파이 브로드캐스팅이 적용
# 즉 train_input에 있는 모든 원소를 제곱한다
# 데이터 셋 크기 확인
print(train_poly.shape, test_poly.shape)

# train_poly를 사용해 선형 회귀 모델을 다시 훈련
# 이 모델이 2차 방정식의 a, b, c를 잘 찾을 것이다
# 2차 방정식 그래프를 찾기 위해 훈련 세트에 제곱 항을 추가했지만 타깃값은 그대로 사용한다
# 목표하는 값은 어떤 그래프를 훈련하든지 바꿀 필요가 없다

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50 ** 2, 50]]))

# 1573 출력

# 모델이 훈련한 계수와 절편 출력
print(lr.coef_, lr.intercept_)

# [1.01... -21.55] 116.05
# 해당 모델은 다음과 같은 그래프를 학습
# 다항식
# 무게 = 1.01 * 길이² - 21.6 * 길이 + 116.05
# 다항식을 사용한 선형 회귀를 다항 회귀라고 부른다

# 검증 산점도
# 짧은 직선을 이어서 그려 곡선 표현
# 1씩 짧게 끊어서 그림

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열 생성 (길이)
point = np.arange(15, 50)

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)

# 15에서 49까지 2차 방정식 그래프를 그린다
plt.plot(point, 1.01 * point ** 2 - 21.6*point + 116.05)

# 50cm 농어 데이터
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# R² 점수 측정
print(lr.score(train_poly, train_target))
# 0.970
print(lr.score(test_poly, test_target))
# 0.977

# 아직 테스트 세트의 점수가 조금 더 높다 
# 과소적합이 남아있다





