# 랜덤박스의 생성의 확률 계산 

# 데이터 준비
# 판다스로 csv 데이터를 읽어 데이터 프레임으로 변환 후 출력
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
# 처음 5개 행 출력
fish.head()

# 데이터프레임은 판다스에서 제공하는 2차원 표 형식의 주요 데이터 구조
# 넘파이 배열과 비슷하게 열과 행으로 이루어져 있다
# 데이터프레임은 통계와 그래프를 위한 메서드를 풍부하게 제공한다 
# 데이터 프레임은 넘파이로 상호 변환이 쉽고 사이킷런과도 잘 호환된다

# 어떤 종류의 생선이 있는지 판다스의 unique() 함수를 이용해 Species 열에서 고유값 추출
print(pd.unique(fish['Species']))

# Species 열을 타깃으로 만들고 나머지 5개 열은 입력 데이터로 사용
# 데이터 프레임에서 열을 선택하는 방법은 원하는 열을 리스트로 나열하면 된다
# to_numpy() 메서드로 넘파이 배열로 변환
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

print(fish_input[:5])

# 타깃 데이터 추출
# fish[['Species']]와 같이 사용하면 2차원 배열이 되기 때문에 조심해야 한다
fish_target = fish['Species'].to_numpy()

# 훈련세트, 테스트세트 생성
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

# 훈련세트, 테스트세트 표준화 전처리
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃 분류기의 확률 예측
# KNeighborsClassifier 클래스 객체를 사용하여 훈련
from sklearn.neighbors import KNeighborsClassifier

kn = KneighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
# 0.89
print(kn.score(test_scaled, test_target))
# 0.85

# 현재 타깃 데이터에 2개 이상의 클래스가 들어있는데 이런 문제를 다중 분류라고 부른다
# 2장에서는 타깃 값을 1과 0으로 만들었지만 문자열로 된 타깃값을 그대로 사용할 수 있다
# 타깃값을 그대로 사이킷런 모델에 전달하면 순서가 자동으로 알파벳 순으로 매겨진다
# 따라서 pd.unique(fish['Species'])로 출력했던 순서와 다르다
# KNeighborsClassifier에서 정렬된 타깃값은 classes_ 속성에 저장
print(kn.classes_)

# predict() 메서드는 타깃값으로 예측을 출력한다 
print(kn.predict(test_scaled[:5]))
# ['Perch', 'Smelt', 'Pike', 'Perch', 'Perch']

# 사이킷런 분류 모델은 predict_proba() 메서드로 클래스별 확률값을 반환
# 넘파이 round() 함수는 기본으로 소수점 첫째 자리에서 반올림을 하는데 decimals 매개변수로 유지할 소수점 아래 자릿수를 지정 가능
import numpy as np

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
#  샘플 -> [ 첫 번째 클래스(Bream)에 대한 확률. 두 번째 클래스(Parkki)에 대한 확률 ...]
# 순서는 print(kn.classes_)에서 출력되는 순으로 지정된다
# [[0.     0.     1.     0.     0.     0.     0.    ]
#  [0.     0.     0.     0.     0.     1.     0.    ]
#  [0.     0.     0.     1.     0.     0.     0.    ]
#  [0.     0.     0.6667 0.     0.3333 0.     0.    ]
#  [0.     0.     0.6667 0.     0.3333 0.     0.    ]]

# 모델이 계산한 확률이 가장 가까운 이웃의 비율 검증
# kneighbors() 메서드의 입력은 2차원 배열이여야 한다
# 넘파이의 슬라이싱 연산자는 하나의 샘플만 선택해도 항상 2차원 배열이 만들어진다

# 4번째 샘플의 이웃 출력
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
# [['Roach' 'Perch' 'Perch']]

# Roach가 1개이고 Perch가 2개다
# 따라서 세 번째 클래스에 대한 확률은 2/3 = 0.6667 이고 다섯번째 클래스에 대한 확률은 1/3 = 0.3333이다

# 로지스틱 회귀(logistic regression)
# 이름은 회귀이지만 분류 모델이다
# 이 알고리즘은 선형 회귀와 동일하게 선형 방정식을 삭습한다
# 예시 z = a * (Weight) + b * (Length) + c * (Diagonal) + d * (Height) + e * (Width) + f
# 여기서에 a,b,c,d,e는 가중치 혹은 계수이다 
# 특성은 늘어났지만 다중 회귀를 위한 선형 방정식과 같다
# z는 어떤 값도 가능하다 하지만 확률이 되려면 0 ~ 1 (또는 0 ~ 100%) 사이 값이 되어야 한다

# z가 아주 큰 음수일 때 0이 되고 아주 큰 양수일 때 1이 되도록 바꾸려면
# 시그모이드 함수(sigmoid function 또는 로지스틱 함수(logistic function))를 사용하면 가능하다
# 선형 방정식의 출력 z의 음수를 사용해 자연 상수 e를 거듭제곱하고 1을 더한 값의 역수를 취한다
# z가 무한하게 큰 음수일 경우 이 함수는 0에 가까워지고 
# z가 무한하게 큰 양수가 될 때는 1에 가까워진다
# z가 0이 될 때는 0.5가 된다
# z가 어떤 값이 되더라도 ø는 절대로 0 ~ 1 사이의 범위를 벗어날 수 없다

# -5와 5가 시그모이드 함수를 사용하면 값이 0 ~ 1 사이의 범위에서 생성되는지 테스트
# 넘파이를 사용하여 시그모이드 함수 그래프 생성
# -5와 5 사이에 0.1 간격으로 배열 z를 만든 다음 z위치마다 시그모이드 함수를 계산
# 지수 함수 계산은 np.exp() 함수를 사용
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()

# 도미와 빙어로 이진 분류 수행
# 시그모이드 함수의 출력이 0.5보다 크면 양성 클래스, 0.5보다 작으면 음성 클래스
# 정확히 0.5일때는 라이브러리마다 다를 수 있다 사이킷런은 0.5일때 음성 클래스로 판단

# 넘파이 배열은 True, False 값을 전달하여 행을 선택할 수 있다 이를 불리언 인덱싱(boolean indexing)이라고 한다
# 불리언 인덱싱 예제
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])
# ['A', 'C']

# 불리언 인덱싱을 사용하여 도미와 빙어의 행만 가져오기
# Bream | Smelt면 True 아니면 False가 담긴 배열 리턴
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀 모델은 LogisticRegression 클래스를 사용하며 선형 모델이므로 sklearn.linear_model 패키지 아래에 있다
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# 예측
print(lr.predict(train_bream_smelt[:5]))
['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']

# KNeighborsClassifier와 마찬가지로 예측 확률은 predict_proba() 메서드에서 제공
print(lr.predict_proba(train_bream_smelt[:5]))
# [[0.99759855 0.00240145]
#  [0.02735183 0.97264817]
#  [0.99486072 0.00513928]
#  [0.98584202 0.01415798]
#  [0.99767269 0.00232731]]

# 첫 번째 열이 음성 클래스(0)에 대한 확률이고 두 번째 열이 양성 클래스(1)에 대한 확률

# k-최근접 이웃 분류기에서 처럼 사이킷런은 타깃값을 알파벳순으로 정렬
# classes_ 속성에서 확인
print(lr.classes_)
# ['Bream' 'Smelt']
# 빙어(Smelt)가 양성 클래스다 
# 만약 도미(Bream)을 양성 클래스로 사용하고 싶다면 Bream의 타깃값을 1로 만들고 나머지 타깃값은 0으로 만들어 사용하면 된다

# 로지스틱 회귀가 학습한 계수 확인
print(lr.coef_, lr.intercept_)
# [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
# 로지스틱 회귀 모델이 학습한 방정식
# # z = -0.404 * (Weight) - 0.576 * (Length) - 0.663 * (Diagonal) - 1.013 * (Height) - 0.732 * (Width) - 2.161

# LogisticRegression 클래스의 decision_function() 메서드로 z값 출력
# train_bream_smelt의 처음 5개 샘플의 z값 출력
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
# [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]

# 이 z값을 시그모이드 함수에 통과시키면 확률을 얻을 수 있다
# 파이썬의 사이파이(scipy)라이브러리에도 시그모이드 함수 expit()가 있다 
# np.exp() 함수를 사용해 분수 계산을 하는 것보다 훨씬 편리하고 안전하다

# decisions 배열의 값을 확률로 변환
from scipy.special import expit
print(expit(decisions))
# [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]
# predict_proba() 메서드 출력의 두 번째 열의 값과 동일
# decision_function() 메서드는 양성 클래스에 대한 z값을 반환

# 이진 분류일 경우 predict_proba() 메서드는 음성 클래스와 양성 클래스에 대한 확률을 출력
# decision_function() 메서드는 양성 클래스에 대한 z값을 계산
# coef_ 속성과 intercept_ 속성에는 로지스틱 모델이 학습한 선형 방정식의 계수가 들어 있다

# LogisticRegression 클래스를 사용해 다중 분류 수행
# LogisticRegression 클래스는 기본적으로 반복적인 알고리즘을 사용 
# max_iter 매개변수에서 반복 횟수를 지정하며 기본값은 100

# LogisticRegression은 기본적으로 릿지 회귀와 같이 계수의 제곱을 규제한다
# 이런 규제는 L2 규제라고도 부른다
# 릿지 회귀에서는 alpha 매개변수로 규제의 양을 조절했다 
# alpha 값이 커지면 규제도 커진다
# LogisticRegression에서 규제를 제어하는 매개변수는 C이다 
# 하지만 C는 alpha와 반대로 작을수록 규제가 커진다 
# C의 기본값은 1

# 로지스틱 회귀 모델 훈련
# 규제 = 20 / 반복 1000
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
# 0.93
print(lr.score(test_scaled, test_target))
# 0.92
# 모두 점수가 높고 과대적합이나 과소적합으로 치우치지 않았다

# 샘플 예측 출력
print(lr.predict(test_scaled[:5]))
# ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']

# 샘플 5개에 대한 예측 확률 출력
# 소수점 네 번째 자리에서 반올림
proba = lr.predict_proba(test_scaled[:5])
print(lr.classes_)
print(np.round(proba, decimals=3))

# 5개의 샘플에 7개의 생선에 대한 확률 계산
# ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
# [[0.    0.014 0.841 0.    0.136 0.007 0.003]
#  [0.    0.003 0.044 0.    0.007 0.946 0.   ]
#  [0.    0.    0.034 0.935 0.015 0.016 0.   ]
#  [0.011 0.034 0.306 0.007 0.567 0.    0.076]
#  [0.    0.    0.904 0.002 0.089 0.002 0.001]]

# 다중 분류의 선형 방정식
print(lr.coef_.shape, lr.intercept_.shape)
# (7, 5) (7,)
# 해당 데이터는 5개의 특성을 사용하므로 coef_배열의 열은 5개다 그런데 행이 7이고 intercept_도 7개다
# 이진 분류에서 보았던 z를 7개나 계산한다는 의미이다
# 다중 분류는 클래스마다 z값을 하나씩 계산한다
# 가장 높은 z 값을 출력하는 클래스가 예측 클래스가 된다

# 이진 분류에서는 시그모이드 함수를 사용해 z를 0과 1사이의 값으로 변환했다
# 다중 분류는 이와 달리 소프트맥스(softmax) 함수를 사용하여 7개의 z 값을 확률로 변환한다

# 소프트맥스 함수
# 시그모이드 함수는 하나의 선형 방정식의 출력값을 0 ~ 1 사이로 압축
# 소프트맥스 함수는 여러 개의 선형 방정식의 출력값을 0 ~ 1 사이로 합축하고 전체 합이 1이 되도록 만든다 
# 이를 위해 지수 함수를 사용하기 때문에 정규화된 지수 함수라고도 부른다

# 소프트맥스 계산 방식
# 7개의 z값을 z1~z7이라고 이름을 붙인다
# z1~z7까지 값을 사용해 지수 함수 eᶻ¹ ~ eᶻ⁷을 계산해 모두 더한다 이를 e_sum이라 정의한다
# e_sum = eᶻ¹ + eᶻ² + eᶻ³ + eᶻ⁴ + eᶻ⁵ + eᶻ⁶ + eᶻ⁷
# 그 다음 eᶻ¹ ~ eᶻ⁷을 각각 e_sum으로 나누어 준다
# s1 = e_sum / eᶻ¹, s2 = e_sum / eᶻ² ...
# s1에서 s7까지 모두 더하면 분자와 분모가 같아지므로 1이 된다 

# 시그모이드 함수와 스프트맥스 함수는 신경망을 배울 때 또다시 쓰인다

# 이진 분류와 마찬가지로 decision_function() 메서드로 z1 ~ z7까지의 값을 구한다
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# [[ -6.5    1.03   5.16  -2.73   3.34   0.33  -0.63]
#  [-10.86   1.93   4.77  -2.4    2.98   7.84  -4.26]
#  [ -4.34  -6.23   3.17   6.49   2.36   2.42  -3.87]
#  [ -0.68   0.45   2.65  -1.19   3.26  -5.75   1.26]
#  [ -6.4   -1.99   5.82  -0.11   3.5   -0.11  -0.71]]

# 그 다음 사이파이의 소프트 맥스 함수를 사용한다 
# scipy.special 아래에 softmax() 함수를 임포트해서 사용
from scipy.special import softmax

proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))

# [[0.    0.014 0.841 0.    0.136 0.007 0.003]
#  [0.    0.003 0.044 0.    0.007 0.946 0.   ]
#  [0.    0.    0.034 0.935 0.015 0.016 0.   ]
#  [0.011 0.034 0.306 0.007 0.567 0.    0.076]
#  [0.    0.    0.904 0.002 0.089 0.002 0.001]]
# 출력 결과가 앞서 구한 proba 배열과 같다

# softmax()의 axis 매개변수는 소프트맥스를 계산할 축을 지정 
# axis=1로 지정하여 각 행, 즉 각 샘플에 대해 소프트맥스를 계산 
# 만약 axis 매개변수를 지정하지 않으면 배열 전체에 대해 소프트맥스를 계산


