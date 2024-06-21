# 판다스로 와인 데이터 불러오기
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 와인 샘플 5개 출력
wine.head()
# 	alcohol	sugar	pH	class
# 0	9.4	1.9	3.51	0.0
# 1	9.8	2.6	3.20	0.0
# 2	9.8	2.3	3.26	0.0
# 3	9.8	1.9	3.16	0.0
# 4	9.4	1.9	3.51	0.0

# 레드와인, 화이트와인 구분은 이진 분류 문제이고 화이트 와인이 양성 클래스
# 즉 전체 와인 데이터에서 화이트 와인을 골라내는 문제

# 판다스의 info() 메서드
# 데이터프레임의 각 열의 데이터 타입과 누락된 데이터가 있는지 확인하는데 유용
wine.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 6497 entries, 0 to 6496
# Data columns (total 4 columns):
#  #   Column   Non-Null Count  Dtype  
# ---  ------   --------------  -----  
#  0   alcohol  6497 non-null   float64
#  1   sugar    6497 non-null   float64
#  2   pH       6497 non-null   float64
#  3   class    6497 non-null   float64
# dtypes: float64(4)
# memory usage: 203.2 KB

# 출력 결과를 보면 총 6497개의 샘플이 있고 4개의 열은 모두 실숫값 
# Non-Null Count가 모두 6497이므로 누락된 값은 없다

# 누락된 값이 있다면 그 데이터를 버리거나 평균값으로 채운 후 사용할 수 있다
# 어떤 방식이 최선인지는 미리 알기 어렵다
# 두 가지 모두 시도해보아야 한다
# 여기에서도 항상 훈련 세트의 통계 값으로 테스트 세트를 변환해야 한다
# 즉 훈련 세트의 평균값으로 테스트 세트의 누락된 값을 채워야 한다

# 판다스의 describe()
# 열에 대한 간략한 통계를 출력한다 (최소, 최대, 평균값 등)
wine.describe()

#         alcohol	sugar	pH	class
# count	6497.000000	6497.000000	6497.000000	6497.000000
# mean	10.491801	5.443235	3.218501	0.753886      평균
# std	    1.192712	4.757804	0.160787	0.430779  표준편차
# min	    8.000000	0.600000	2.720000	0.000000  최소
# 25%	    9.500000	1.800000	3.110000	1.000000  1사분위수
# 50%	    10.300000	3.000000	3.210000	1.000000  중간값 / 2사분위수
# 75%	    11.300000	8.100000	3.320000	1.000000  3사분위수
# max	    14.900000	65.800000	4.010000	1.000000  최대

# 사분위수는 데이터를 순서대로 4등분 한 값이다 
# 예를 들어 2사분위수(중간값)는 데이터를 일렬로 늘어놓았을 때 정중앙의 값이다
# 만약 데이터 개수가 짝수개라 중앙값을 선택할 수 없다면 가운데 2개의 값의 평균을 사용한다

# 여기서 알 수 있는 것은 알코올 도수와 당도, PH 값의 스케일이 다르다는 것이다
# 사이킷런의 StandardScaler 클래스를 사용해 특성을 표준화하는 전처리 작업이 필요하다

# 판다스 데이터프레임을 넘파이 배열로 바꾸고 훈련 세트와 테스트 세트로 나눈다

# 타깃 데이터와 샘플 데이터 분리
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split

# test_size를 지정하지 않으면 기본값으로 25%를 테스트 세트로 지정 
# 샘플이 충분히 많으므로 0.2로 지정하여 20%정도만 테스트 세트로 나눈다
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# 훈련 세트와 테스트 세트의 크기 확인
print(train_input.shape, test_input.shape)
# (5197, 3) (1300, 3)

# StandardScaler 클래스를 사용해 훈련 세트를 전처리
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
# 0.7808350971714451
# 0.7776923076923077

# 모델이 다소 과소적합되었다
# 이 문제를 해결하기 위해 규제 매개변수 C의 값을 바꿀수도 있고 solver 매개변수에서 다른 알고리즘을 선택할 수도 있다
# 또는 다항 특성을 만들어 추가할 수도 있다

# 로지스틱 회귀가 학습한 계수와 절편을 출력
print(lr.coef_, lr.intercept_)

# [[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]

# 결정트리
# 결정 트리 모델은 스무고개와 같다 
# 질문을 하나씩 던져서 정답과 맞춰간다
# 데이터를 잘 나눌 수 있는 질문을 찾는다면 계속 질문을 추가해서 분류 정확도를 높일 수 있다

# 사이킷런의 DecisionTreeClassifier 클래스를 사용해 결정 트리 모델을 사용할 수 있다
# 사이킷런의 결정 트리 알고리즘은 노드에서 최적의 분할을 찾기 전에 특성의 순서를 섞는다
# 따라서 약간의 무작위성이 주입되는데 실행할 때마다 점수가 조금씩 달라질 수 있기 때문이다
# 실전에서는 필요하지 않다

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
# 0.996921300750433
print(dt.score(test_scaled, test_target))
# 0.8592307692307692

# 훈련세트는 점수가 매우 높지만 테스트 세트의 성능은 그에 비해 조금 낮다
# 과대적합된 모델이다 

# 이 모델을 그림으로 표현하려면 plot_tree() 함수를 사용하면 된다
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()

# 노드는 결정 트리를 구성하는 핵심 요소이다 
# 노드는 훈련 데이터의 특성에 대한 테스트를 표현한다
# 예를들어 현재 샘플의 당도가 -0.239보다 작거나 같은지 테스트한다 
# 가지(branch)는 테스트의 결과(True, False)를 나타내며 일반적으로 하나의 노드는 2개의 가지를 가진다
# 걀정 트리의 맨 위 노드를 루트노드라 부르고 맨 아래 끝에 달린 노드를 리프 노드라 한다

# plot_tree() 함수에서 
# max_depth 매개변수를 1로 주면 루트 노드를 제외하고 하나의 노드를 더 확장하여 그린다
# filed 매개변수에서 클래스에 맞게 노드의 색을 칠할 수 있다
# feature_names 매개변수에는 특성의 이름을 전달할 수 있다

plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'suger', 'pH'])
plt.show()

# 루트 노드는 당도(sugar)가 -0.239 이하인지 질문한다
# 만약 어떤 샘플의 당도가 -0.239와 같거나 작으면 왼쪽 가지로 간다 그렇지 않으면 오른쪽 가지로 간다
# 즉 왼쪽이 Yes, 오른쪽이 No다 
# 루트 노드의 총 샘플 수(samples)는 5197개이다 
# 이 중에서 음성 클래스(레드 와인)는 1258개이고 양성 클래스(화이트 와인)는 3939개이다
# 이 값이 value에 나타나 있다

# 결정 트리에서 예측하는 방법은 간단하다
# 리프 노드에서 가장 많은 클래스가 예측 클래스가 된다 
# k-최근접 이웃과 매우 비슷하다

# 만약 결정 트리를 회귀 문제에 적용하면 리프 노드에 도달한 샘플의 타깃을 평균하여 
# 예측값으로 사용 
# 사이킷런의 결정 트리 회귀 모델은 DecisionTreeRegressor이다

# 불순도
# 트리를 그려보면 노드 안에 gini라는 값이 있는데 이 값은 지니 불순도(Gini impurity)를 의미
# DecisionTreeClassifier 클래스의 criterion 매개변수의 기본값이 gini이다 
# criterion 매개변수의 용도는 노드에서 데이터를 분할할 기준을 정하는 것이다
# 앞에서 그린 트리에서 루트 노드는 criterion 매개변수에 지정한 지니 불순도를 사용하여 당도 -0.239를 기준으로 왼쪽, 오른쪽 노드로 나눈다

# 지니 불순도 계산
# 클래스의 비율을 제곱해서 더한 다음 1에서 뺀다
# 지니 불순도 = 1 - (음성 클래스 비율² + 양성 클래스 비율²)

# 다중 클래스 문제라면 클래스가 더 많겠지만 계산방법은 동일

# 예제에서 출력한 루트 노드의 지니 불순도 계산
# 루트 노드는 5197개의 샘플이 있고 그 중에 1258개가 음성 클래스 3939개가 양성 클래스이다
# 1- ((1258 / 5197)² + (3939 / 5197)²) = 0.367

# 만약 100개의 샘플이 있는 어떤 노드의 두 클래스의 비율이 정확이 1/2씩이라면 지니 불순도는 0.5가 되어 최악이 된다
# 1 - ((50 / 100)² + (50 / 100)²) = 0.5

# 노드에 하나의 클래스만 있다면 지니 불순도는 0이 되어 가장 작다 이런 노드를 순수 노드라고도 부른다
# 1 - ((0 / 100)² + (100 / 100)²) = 0

# 결정 트리 모델은 부모 노드와 자식 노드의 불순도 차이가 가능한 크도록 트리를 성장시킨다
# 부모 노드와 자식 노드의 불순도 차이 계산 방법
# 자식 노드의 불순도를 샘플 개수에 비례하여 모두 더한다 
# 그 다음 부모 노드의 불순도에서 뺀다

# 예제의 부모 노드와 자식 노드의 불순도 차이 계산
# 왼쪽 노드로는 2922개의 샘플이 이동했고 오른쪽 노드로는 2275개의 샘플이 이동
# 부모의 불순도 - (왼쪽 노드 샘플 수 / 부모의 샘플 수) * 왼쪽 노드 불순도 - (오른쪽 노드 샘플 수 / 부모의 샘플 수) * 오른쪽 노드 불순도
# 0.367 - (2922 / 5197) * 0.481 - (2275 / 5197) * 0.069 = 0.066

# 부모와 자식 노드 사이의 불순도 차이를 정보 이득(information gain)이라고 부른다
# 이 알고리즘은 정보 이득이 최대가 되도록 데이터를 나눈다
# 이때 지니 불순도를 기준으로 사용한다

# DecisionTreeClassifier 클래스에서 criterion='entropy'를 지정하여 엔트로피 불순도를 사용할 수 있다
# 엔트로피 불순도도 노드의 클래스 비율을 사용하지만 지니 불순도처럼 제곱이 아니라 밑이 2인 로그를 사용하여 곱한다
# 예를 들어 루트 노드의 엔트로피 불순도는 다음과 같다
# -음성 클래스 비율 * log₂(음성 클래스 비율) - 양성 클래스 비율 * log₂(양성 클래스 비율)
# -(1258 / 5197) * log₂(1258 / 5197) - (3939 / 5197) * log₂(3939 / 5197) = 0.798