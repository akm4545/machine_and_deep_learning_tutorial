# 1개의 특성을 사용했을 때 선형 회귀 모델이 학습하는 것은 직선이다
# 2개의 특성을 사용하면 선형 회귀는 평면을 학습한다
# 특성이 2개면 타깃값과 함께 3차원 공간을 형성하고 선형 회귀 방정식 
# 타깃 = a * 특성1 + b * 특성2 + 절편은 평면이 된다
# 선형 회귀를 단순한 직선이나 평면으로 생각하여 성능이 무조건 낮다고 오해해서는 안된다
# 특성이 많은 고차원에서는 선형 회귀가 매우 복잡한 모델을 표현할 수 있다

# 이번에는 농어의 길이뿐만 아니라 농어의 높이와 두께도 함께 사용해서 학습
# 이전 절에서처럼 3개의 특성을 각각 제곱하여 추가
# 거기다가 각 특성을 서로 곱해서 또 다른 특성을 만든다 (농어 길어 * 농어 높이)
# 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업을 특성 공학(feature engineering) 이라고 부른다

# 판다스(pandas)는 유명한 데이터 분석 라이브러리다 
# 데이터프레임(dataframe)은 판다스의 핵심 데이터 구조
# 넘파이 배열과 비슷하게 다차원 배열을 다룰 수 있지만 훨씬 더 많은 기능을 제공 
# 데이터프레임은 넘파이 배열로 쉽게 바꿀 수 있다

# 판다스 데이터프레임을 만들기 위해 많이 사용하는 파일은 csv 파일이다
# 판다스의 read_csv() 함수에 주소를 넣으면 데이터를 읽어올 수 있다
# 그 뒤 to_numpy() 메서드를 사용해 넘파이 배열로 바꾼다
import pandas as pd # pd는 관례적으로 사용하는 판다스의 별칭

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)

# 타깃 데이터 
import numpy as np

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# perch_full과 perch_weight를 훈련 세트와 테스트 세트로 나눈다
from sklean.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42
)

# 사이킷런은 특성을 만들거나 전처리하기 위한 다양한 클래스를 제공한다
# 사이킷런에서는 이런 클래스를 변환기(transformer)
# 사이킷런의 모델 클래스에 일관된 fit(), score(), predict() 메서드가 있는 것처럼 변환기 클래스는 모두 fit(), transform() 메서드를 제공

# LinearRegression 같은 사이킷런의 모델 클래스는 추정기(estimator) 라고도 부른다

# PolynomialFeatures 변환기 클래스 사용
# sklearn.preprocessing 패키지에 포함
from sklearn.preprocessing import PolynomialFeatures

# 2개의 특성 2와 3으로 이루어진 샘플 하나를 적용
# 클래스의 객체를 만든 다음 fit(), transform() 메서드 차례로 호출
poly = PolynomialFeatures()

# 훈련(fit)을 해야 변환이 가능하다 사이킷런의 일관된 api 떄문에 두 단계로 나뉘어져 있다
# 두 메서드를 하나로 붙인 fit_transform 메서드도 존재
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
# [1. 2. 3. 4. 6. 9.] 출력

# fit() 메서드는 새롭게 만들 특성 조합을 찾고 transform() 메서드는 실제로 데이터를 변환
# 변환기는 입력 데이터를 변환하는 데 타깃 데이터가 필요하지 않다
# 따라서 모델 클래스와는 다르게 fit() 메서드에 입력 데이터만 전달

# PolynomialFeatures 클래스는 기본적으로 각 특성을 제곱한 항을 추가하고 특성끼리 서로 곱한 항을 추가한다
# 2와 3을 각기 제곱한 4와 9가 추가되었고 2와 3을 곱한 6이 추가
# 무게 = a * 길이 + b * 높이 + c * 두께 + d * 1
# 선형 방정식의 절편을 항상 값이 1인 특성과 곱해지는 계수라고 볼 수 있다 
# 이로 인해 결괏값에 1이 추가되었다
# 사이킷런의 선형 모델은 자동으로 절편을 추가하므로 굳이 이렇게 특성을 만들 필요가 없다
# include_bias=False로 지정하여 다시 특성 변환

poly = PolynomialFeatures(include_bias=False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
# [2. 3. 4. 6. 9.] 출력
# 절편을 위한 항이 제거되고 특성의 제곱과 특성끼리 곱한 항만 추가
# include_bias=False로 지정하지 않아도 사이킷런 모델은 자동으로 특성에 추가된 절편 항을 무시

# 이 방식으로 train_input에 적용
poly = PolynomialFeatures(include_bias=False)
# 훈련
poly.fit(train_input)
# 특성을 만든다
train_poly = poly.transform(train_input)
print(train_poly.shape)
# (42, 9) = 9개의 특성을 가진 42개의 데이터 생성됨

# PolynomialFeatures 클래스의 get_feature_names_out() 메서드를 호출하여 특성이 어떤 입력 조합으로 만들어졌는지 출력
poly.get_feature_names_out()
# ['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2']

# x0은 첫 번째 특성을 의미하고 x0^2는 첫 번째 특성의 제곱 x0 x1은 첫 번째 특성과 두 번째 특성의 곱을 나타내는 식

# 테스트 세트 변환
test_poly = poly.transform(test_input)

# PolynomialFeatures 클래스는 fit() 메서드에서 만들 특성의 조합을 준비하기만 하고 별도의 통계 값을 구하지는 않는다
# 따라서 테스트 세트를 따로 변환해도 된다
# 항상 훈련 세트를 기준으로 테스트 세트를 변환하는 습관을 들이는 것이 좋다

# 다중 회귀 모델을 훈련하는 것은 선형 회귀 모델을 훈련하는 것과 같다.
# 여러 개의 특성을 사용하여 선형 회귀를 수행하는 것 뿐이다

# 다중 회귀 모델 훈련
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.score(train_poly, train_target))
# 0.99... 점수가 나온다
# 특성이 늘어나면 선형 회귀의 능력은 매우 강력해진다
print(lr.score(test_poly, test_target))
# 0.97... 
# 테스트 세트에 대한 점수는 높아지지 않았지만 농어의 길이만 사용했을 때 있던 과소적합 문제는 더 이상 나타나지 않는다

# PolynomialFeatures 클래스의 defree 매개변수를 사용하여 필요한 고차항의 최대 차수를 지정할 수 있다
# 5 제곱까지 특성을 만들어 출력
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape)
# (42, 55) 만들어진 특성의 개수가 55개가 나온다

# 선형 회귀 모델 훈련
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
# 0.999999... 

# 테스트 세트 점수 출력
print(lr.score(test_poly, test_target))
# -144.40...

# 특성의 개수를 늘리면 선형 모델은 아주 강력해진다
# 훈련 세트에 대해 거의 완벽하게 학습할 수 있다
# 하지만 이런 모델은 훈련 세트에 너무 과대적합되므로 테스트 세트에서는 형편없는 점수를 만든다

# 샘플의 개수보다 특성의 개수가 많은 데이터로 훈련하면 완벽하게 학습할 수 있는 것이 당연하다
# 예를 들어 42개의 참새를 맞추기 위해 딱 한 번 새총을 쏴야 한다면 참새 떼 중앙을 겨냥하여 가능한 한 맞출 가능성을 높여야 한다
# 하지만 55번이나 쏠 수 있다면 한 번에 하나씩 모든 참새를 맞출 수 있다

# 규제(regularization)
# 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것
# 모델이 훈련 세트에 과대적합되지 않도록 만드는 것이다
# 선형 회귀 모델의 경우 특성에 곱해지는 계수(또는 기울기)의 크기를 작게 만드는 일이다

# 특성의 스케일(특성의 범위)가 정규화 되지 않으면 여기에 곱해지는 계수 값도 차이가 나게 된다
# 일반적으로 선형 회귀 모델에 규제를 적용할 때 계수 값의 크기가 서로 많이 다르면 공정하게 제어되지 않는다
# 사이킷런의 StandardScaler 클래스는 변환기의 하나로 스케일의 정규화를 도와준다 (평균과 표준편차를 구해 특성을 표준점수로 바꿈)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
# 객체를 훈련
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# StandardScaler 클래스 객체의 mean_, scale_ 속성에 훈련 세트에서 학습한 평균과 표준편차가 저장
# 특성마다 계산하므로 위의 예제 코드에서는 55개의 평균과 표준 편차가 들어있다

# 릿지 회귀
# 선형 회귀 모델에 규제를 추가한 모델을 릿지(ridge)와 라쏘(lasso)라고 부른다
# 두 모델은 규제를 가하는 방법이 다르다
# 릿지는 계수를 제곱한 값을 기준으로 규제를 적용하고 라쏘는 계수의 절댓값을 기준으로 규제를 적용
# 일반적으로 릿지를 조금 더 선호
# 두 알로리즘 모두 계수의 크기를 줄이지만 라쏘는 아예 0으로 만들 수도 있다 

# 릿지와 라쏘 모두 sklearn.linear_model 패키지 안에 있다
# 릿지 모델 훈련
from sklean.linear_model import Ridge

ridge = Ridge()
ridge.fit(train_scaled, train_target)

# 훈련 세트 점수 출력
print(ridge.score(train_scaled, train_target))
# 0.989...

# 테스트 세트 점수 출력
print(ridge.score(test_scaled, test_target))
# 0.979...

# 많은 특성을 사용했음에도 불구하고 훈련 세트에 너무 과대적합되지 않아 테스트 세트에서도 좋은 성능을 내고 있다

# 릿지와 라쏘 모델을 사용할 때 규제의 양을 임의로 조절할 수 있다
# 모델 객체를 만들 때 alpha 매개변수로 규제의 강도를 조절한다 
# alpha 값이 크면 규제 강도가 세지므로 계수 값을 더 줄이고 조금 더 과소적합되도록 유도
# alpha 값이 작으면 계수를 줄이는 역할이 줄어들고 선형 회귀 모델과 유사해지므로 과대적합될 가능성이 크다

# 이렇게 머신러닝 모델이 학습할 수 없고 사람이 알려줘야 하는 파라미터를 하이퍼파라미터(hyperparameter)라고 부른다
# 사이킷런과 같은 머신러닝 라이브러리에서 하이퍼파라미터는 클래스와 메서드의 매개변수로 표현

# 적절한 alpha 값을 찾는 한 가지 방법은 alpha 값에 대한 R² 값의 그래프를 그려 보는 것이다
# 훈련 세트와 테스트 세트의 점수가 가장 가까운 지점이 최적의 alpha 값이 된다

# 맷플롯립을 입포트 후 alpha 값을 바꿀 때마다 score() 메서드의 결과를 저장할 리스트 생성
import matplotlib.pyplot as plt

train_score = []
test_score = []

# alpha 값을 0.001dptj 100까지 10배씩 늘려가며 릿지 회귀 모델을 훈련한 다음 훈련 세트와 테스트 세트의 점수를 파이썬 리스트에 저장
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    # 릿지 모델 생성
    ridge = Ridge(alpha=alpha)
    # 릿지 모델 훈련
    ridge.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수 저장
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# alpha 값을 0.001 부터 10배씩 늘렸기 때문에 이대로 그래프를 그리면 그래프 왼쪽이 너무 촘촘해진다
# alpha_list에 있는 6개의 값을 동일한 간격으로 나타내기 위해 로그 함수로 바꾸어 지수로 표현해야 한다
# 즉 0.001은 -3, 0.01은 -2가 되는 식이다

# 넘파이 로그 함수는 np.log() 와 np.log10()이 있다 
# 전자는 자연 상두 e를 밑으로 하는 자연로그 / 후자는 10을 밑으로 하는 상용로그

plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()