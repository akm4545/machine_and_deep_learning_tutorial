# 정형 데이터(structured data)
# 어떤 구조로 되어 있는 데이터
# 이런 데이터는 csv나 데이터베이스 혹은 엑셀에 저장하기 쉽다
# 프로그래머가 다루는 대부분의 데이터가 정형데이터이다

# 비정형 데이터(unstructured data)
# 데이터베이스나 엑셀로 표현하기 어려운 것들
# 사진, 음악 등이 있다

# 앙상블 학습(ensemble learning)
# 정형 데이터를 다루는데 가장 뛰어난 성과를 내는 알고리즘
# 대부누 결정 트리를 기반으로 만들어져 있다

# 랜덤 포레스트(Random Forest)
# 앙상블 학습의 대표 주자 중 하나로 안정적인 성능을 낸다
# 결정 트리를 랜덤하게 만들어 결정 트리(나무)의 숲을 만든다
# 각 결정 트리의 예측을 사용해 최종 예측을 만든다

# 각 노드를 분할할 때 전체 특성 중에서 일부 특성을 무작위로 고른 다음 이 중에서 최선의 분할을 찾는다
# 분류 모델인 RandomForestClassifier는 기본적으로 전체 특성 개수의 제곱근만큼의 특성을 선택한다
# 즉 4개의 특성이 있다면 노드마다 2개를 랜덤하게 선택하여 사용한다

# 회귀 모델인 RandomForestRegressor는 전체 특성을 사용한다

# 사이킷런의 랜덤 포레스트는 기본적으로 100개의 결정 트리를 이런 방식으로 훈련한다
# 그다음 분류일 떄는 각 트리의 클래스별 확률을 평균하여 가장 높은 확률을 가진 클래스를 예측으로 삼는다
# 회귀일 때는 단순히 각 트리의 예측을 평균한다

# 랜덤 포레스트는 랜덤하게 선택한 샘플과 특성을 사용하기 떄문에 훈련 세트에 과대적합되는 것을 막아주고
# 검증 세트와 테스트 세트에서 안정적인 성능을 얻을 수 있다
# 종종 기본 매개변수 설정만으로도 아주 좋은 결과를 낸다

# 부트스트랩 샘플(bootstrap sample)
# 데이터 세트에서 중복을 허용하여 데이터를 샘플링하는 방식을 부트스트랩 방식이라고 한다
# 랜덤 포레스트는 각 트리를 훈련하기 위한 데이터를 랜덤하게 만든다
# 입력한 훈련 데이터에서 랜더마게 샘플을 추출하여 훈련 데이터를 만든다
# 이때 한 샘플이 중복되어 추출될 수도 있다

# 예를 들어 1000개의 샘플이 들어있는 가방에서 100개의 샘플을 뽑는다면 
# 먼저 1개를 뽑고 뽑았던 1개를 다시 가방에 넣는다
# 이런 식으로 계속해서 100개를 가방에서 뽑으면 중복된 샘플을 뽑을 수 있다

# 기본적으로 부트스트랩 샘플은 훈련 세트의 크기와 같게 만든다

# 분류 = 샘플을 몇 개의 클래스 중 하나로 분류
# 회귀 = 임의의 어떤 숫자를 에측하는 문제

# RandomForestClassifier 클래스를 사용하여 와인 분류
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# cross_validate() 함수를 사용하여 교차 검증 수행
# RandomForestClassifier는 기본적으로 100개의 결정 트리를 사용하므로 n_jobs 매개변수를 -1로 지정하여 
# 모든 CPU 코어를 사용하는 것이 좋다

# cross_validate() 함수의 n_jobs 매개변수도 -1로 지정하여 최대한 병렬로 교차 검증 수행
# return_train_score 매개변수를 True로 지정하면 검증 점수뿐만 아니라 훈련 세트에 대한 점수도 같이 반환
# 훈련 세트와 검증 세트의 점수를 비교하면 과대적합을 파악하는데 용이(return_train_score 매개변수의 기본값은 False)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9973541965122431 0.8905151032797809
# 훈련 세트에 다소 과대적합되어 있다
# 이 예제는 매우 간단하고 특성이 많지 않아 그리드 서치를 사용하더라도 하이퍼파라미터 튜닝의 결과가 크게 나아지지 않는다

# 랜덤 포레스트는 결정 트리의 앙상블이기 때문에 DecisionTreeClassifier가 제공하는 매개변수를 모두 제공
# criterion, max_depth, max_features, min_samples_split, min_impurity_decrease, min_samples_leaf 등이다
# 또한 결정 트리의 큰 장점 중 하나인 특성 중요도를 계산
# 랜덤 포레스트의 특성 중요도는 각 결정 트리의 특성 중요도를 취합한 것이다

# 랜덤 포레스트 모델을 훈련 세트에 훈련한 후 특성 중요도를 출력
rf.fit(train_input, train_target)
print(rf.feature_importances_)
# [0.23167441 0.50039841 0.26792718]

# 결정 트리에서 만든 특성 중요도와 다른 이유는 랜덤 포레스트가 특성의 일부를 랜덤하게 선택하여
# 결정 트리를 훈련하기 때문

# 그 결과 하나의 특성에 과도하게 집중하지 않고 좀 더 많은 특성이 훈련에 기여할 기회를 얻는다
# 이는 과대적합을 줄이고 일반화 성능을 높이는데 도움이 된다

# RandomForestClassifier는 자체적으로 모델을 평가하는 점수를 얻을 수 있다
# 랜덤 포레스트는 훈련 세트에서 중복을 허용하여 부트스트랩 샘플을 만들어 결정 트리를 훈련하는데 
# 이때 부트스트랩 샘플에 포함되지 않고 남는 샘플이 있다
# 이런 샘플을 OOB(out of bag) 샘플이라고 한다
# 이 남는 샘플을 사용하여 부트스트랩 샘플로 훈련한 결정 트리를 평가할 수 있다
# 마치 검증 세트의 역할을 한다

# 이 점수를 얻으려면 RandomForestClassifier 클래스의 oob_score 매개변수를 True로 지정해야 한다 (기본값 False)
# 이렇게 하면 랜덤 포레스트는 각 결정 트리의 OOB 점수를 평균하여 출력한다

# OOB 점수 출력
rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)

print(rf.oob_score_)
# 0.8934000384837406

# OOB 점수를 사용하면 교차 검증을 대신할 수 있어서 결과적으로 훈련 세트에 더 많은 샘플을 사용할 수 있다

# 엑스트라 트리(Extra Trees)
# 랜덤 포레스트와 매우 비슷하게 동작
# 기본적으로 100개의 결정 트리를 훈련한다
# 랜덤 포레스트와 동일하게 결정 트리가 제공하는 대부분의 매개변수를 지원한다
# 또한 전체 특성 중에 일부 특성을 랜덤하게 선택하여 노드를 분할하는 데 사용한다

# 랜덤 포레스트와 엑스트라 트리의 차이점은 부트스트랩 샘플을 사용하지 않는다는 점이다
# 각 결정 트리를 만들 때 전체 훈련 세트를 사용한다
# 대신 노드를 분할할 때 가장 좋은 분할을 찾는 것이 아니라 무작위로 분할한다

# DecisionTreeClasifier의 splitter 매개변수를 random으로 지정한 결정 트리를 사용한다

# 하나의 결정 트리에서 특성을 무작위로 분할한다면 성능이 낮아지겠지만 많은 트리를 앙상블 하기 떄문에
# 과대적합을 막고 검증 세트의 점수를 높이는 효과가 있다

# 사이킷런에서 제공하는 엑스트라 트리는 ExtraTreesClassifier 이다

# 엑스트라 트리 모델의 교차 검증 점수 확인
from sklearn.ensemble import ExtraTreesClassifier

et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score'], np.mean(scores['test_score'])))
# 0.9974503966084433 0.8887848893166506

# 특성이 많지 않아 해당 예제에서는 랜덤 포레스트와 비슷한 결과를 얻는다
# 보통 엑스트라 트리가 무작위성이 좀 더 크기 떄문에 랜덤 포레스트보다 더 많은 결정 트리를 훈련해야 한다
# 하지만 랜덤하게 노드를 분할하기 떄문에 빠른 계산 속도가 엑스트라 트리의 장점이다
# 결정 트리는 최적의 분할을 찾는 데 시간을 많이 소모한다
# 특히 고려해야 할 특성의 개수가 많을 때 더 그렇다
# 만약 무작위로 나눈다면 훨씬 빨리 트리를 구성할 수 있다

# 엑스트라 트리의 특성 중요도 출력
et.fit(train_input, train_target)
print(et.feature_importances_)
# [0.20183568 0.52242907 0.27573525]

# 엑스트라 트리의 회구 버전은 ExtraTreesRegressor 클래스이다


