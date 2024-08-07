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

# 그레이디언트 부스팅(gradient boosting)
# 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블 하는 방법
# 사이킷런의 GradientBoostingClassifier는 기본적으로 깊이가 3인 결정 트리 100개를 사용
# 깊이가 얕은 결정 트리를 사용하기 때문에 과대적합에 강하고 일반적으로 높은 일반화 성능을 기대할 수 있다

# 경사 하강법을 사용하여 트리를 앙상블에 추가한다
# 분류에서는 로지스틱 손실 함수를 사용하고 회귀에서는 평균 제곱 오차 함수를 사용한다

# 손실 함수를 산으로 정의하면 가장 낮은 곳을 찾아 내려오는 방법은 모델의 가중치와 절편을 조금씩 바꾸는 것이다
# 그레이디언트 부스팅은 결정 트리를 계속 추가하면서 가장 낮은 곳을 찾아 이동한다
# 그래서 깊이가 얕은 트리를 사용한다
# 학습률 매개변수로 속도를 조절한다

# GradientBoostingClassifier를 사용하여 와인 데이터셋의 교차 검증 점수 확인
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.8881086892152563 0.8720430147331015

# 거의 과대적합이 되지 않는다
# 그레이디언트 부스팅은 결정 트리의 개수를 늘려도 과대적합에 매우 강하다
# 학습률을 증가시키고 트리의 개수를 늘리면 조금 더 성능이 향상될 수 있다

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9464595437171814 0.8780082549788999

# 결정 트리 개수를 500개로 5배 늘렸지만 과대적합을 잘 억제하고 있다
# 학습률 learning_rate의 기본값은 0.1dlek 
# 그레이디언트 부스팅도 특성 중요도를 제공한다
gb.fit(train_input, train_target)
print(gb.feature_importances_)
# [0.15872278 0.68010884 0.16116839]

# 그레이디언트 부스팅이 랜덤 포레스트보다 일부 특성(당도)에 더 집중한다

# 트리 훈련에 사용할 훈련 세트의 비율을 정하는 subsample 매개변수가 있다
# 이 매개변수의 기본값은 1.0으로 전체 훈련 세트를 사용한다
# subsample이 1보다 작으면 훈련 세트의 일부를 사용한다
# 이는 마치 경사 하강법 단계마다 일부 샘플을 랜덤하게 선택하여 진행하는 확률적 경사 하강법이나 미니채비 경사 하강법과 비슷하다

# 일반적으로 그레이디언트 부스팅이 랜덤 포레스트보다 조금 더 높은 성능을 얻을 수 있다.
# 하지만 순서대로 트리를 추가하기 떄문에 훈련 속도가 느리다 
# GradientBoostingClassifier에는 n_jobs 매개변수가 없다
# 그레이디언트 부스팅의 회귀 버전은 GradientBoostingRegressor이다

# 히스토그램 기반 그레이디언트 부스팅(Histogram-based Gradient Boosting)
# 정형 데이터를 다루는 머신러닝 알고리즘 중에 가장 인기가 높은 알고리즘
# 히스토그램 기반 그레이디언트 부스팅은 먼저 입력 특성을 256개의 구간으로 나눈다
# 따라서 노드를 분할할 때 최적의 분할을 매우 빠르게 찾을 수 있다
# 히스토그램 기반 그레이디언트 부스팅은 256개의 구간 중에서 하나를 떼어 놓고 누락된 값을 위해서 사용한다
# 따라서 입력에 누락된 특성이 있더라도 이를 따로 전처리할 필요가 없다

# 사이킷런의 히스토그램 기반 그레이디언트 부스팅 클래스는 HistGradientBoostingClassifier이다
# 일반적으로 HistGradientBoostingClassifier는 기본 매개변수에서 안정적인 성능을 얻을 수 있다
# 트리의 개수를 지정하는데 n_estimators 대신에 부스팅 반복 횟수를 지정하는 max_iter를 사용한다
# 성능을 높이려면 max_iter 매개변수를 테스트하자 

# HistGradientBoostingClassifier을 사용하여 와인 데이터셋의 교차 검증 점수 확인
# from sklearn import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target, return_train_score=True)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9321723946453317 0.8801241948619236

# 과대적합을 잘 억제하면서 그레이디언트 부스팅보다 조금 더 높은 성능을 제공한다

# 히스토그램 기반 그레이디언트 부스팅의 특성 중요도를 계산하기 위해 permutation_importance() 함수를 사용한다
# 이 함수는 특성을 하나씩 랜덤하게 섞어서 모델의 성능이 변화하는지를 관찰하여 어떤 특성이 중요한지를 계산한다
# 훈련 세트뿐만 아니라 테스트 세트에도 적용할 수 있고 사이킷런에서 제공하는 추정기 모델에 모두 사용할 수 있다

# 히스토그램 기반 그레이디언트 부스팅 모델을 훈련하고 훈련 세트에서 특성 중요도 계산 
# n_repeats 매개변수는 랜덤하게 섞을 횟수를 지정 기본값은 5
from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)

print(result.importances_mean)
# [0.08876275 0.23438522 0.08027708]

# permutation_importance() 함수가 반환하는 객체는 반복하여 얻은 특성 중요도(importances), 평균(importances_mean),
# 표준 편차(importances_std)를 담고 있다
# 평균을 출력해 보면 랜덤 포레스트와 비슷한 비율임을 알 수 있다

# 테스트 세트에서 특성 중요도를 계산
result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)

# [0.05969231 0.20238462 0.049     ]

# 테스트 세트의 결과를 보면 그레이디언트 부스팅과 비슷하게 조금 더 당도에 집중하고 있다
# 이런 분석을 통해 모델을 실전에 투입헀을 때 어떤 특성에 관심을 둘지 예상할 수 있다

# HisGradientBoostingClassifier를 사용해 테스트 세트에서의 성능을 최종적으로 확인
hgb.score(test_input, test_target)
# 0.8723076923076923

# 테스트 세트에서는 약 87%의 정확도를 얻었다
# 실전에 투입하면 성능은 이보다는 조금 더 낮을 것이다
# 앙상블 모델은 확실히 단일 결정 트리보다 좋은 결과를 얻을 수 있다

# 히스토그램 기반 그레이디언트 부스팅의 회귀 버전은 HistGradientBoostingRefressor 클래스에 구현되어 있다
# 사이킷런에서 제공하는 히스토그램 기반 그레이디언트 부스팅이 비교적 새로운 기능이다
# 사이킷런 말고도 그레이디언트 부스팅 알고리즘을 구현한 라이브러리가 ㅇ럿 있다

# 가장 대표적인 라이브러리는 XGBoost이다 
# 이 라이브러리도 코랩에서 사용할 수 있을 뿐만 아니라 사이킷런의 cross_validate() 함수와 함께 사용할 수도 있다
# XGBoost는 다양한 부스팅 알고리즘을 지원한다
# tree_method 매개변수를 hist로 지정하면 히스토그램 기반 그레이디언트 부스팅을 사용할 수 있다

# XGBoost를 사용해 와인 데이터의 교차 검증 점수를 확인
from xgboost import XGBClassifier

xgb = XGBClassifier(tree_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.9558403027491312 0.8782000074035686

# 널리 사용하는 또 다른 히스토그램 기반 그레이디언트 부스팅 라이브러리는 마이크로소프트에서 만든 LightGBM이다
# LightGBM은 빠르고 최신 기술을 많이 적용하고 있어 인기가 점점 높아지고 있다 

from lightgbm import LGBMClassifier

lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, return_train_score=True, n_jobs=-1)

print(np.mean(scores['train_score']), np.mean(scores['test_score']))
# 0.935828414851749 0.8801251203079884

# 사이킷런의 히스토그램 기반 그레이디언트 부스팅이 LightGBM에서 영향을 많이 받았다

