# 테스트 세트를 사용해 자꾸 성능을 확인하다 보면 점점 테스트 세트에 맞추게 되는 셈이다
# 테스트 세트로 일반화 성능을 올바르게 예측하려면 가능한 한 테스트 세트를 사용하지 말아야 한다
# 모델을 만들고 나서 마지막에 딱 한 번만 사용하는 것이 좋다

# 검증 세트
# 테스트 세트를 사용하지 않으면 모델이 과대적합인지 과소적합인지 판단하기 어렵다
# 테스트 세트를 사용하지 않고 이를 측정하는 간단한 방법은 훈련 세트를 또 나누는 것이다
# 이 데이터를 검증 세트(validation set)라고 부른다

# 실제로 많이 사용하는 방법이다
# 훈련 세트에서 다시 20%를 떼어 내어 검증 세트로 만든다

# 보통 20~30%를 테스트 세트와 검증 세트로 떼어 놓는다 하지만 문제에 따라 다르다
# 훈련 데이터가 아주 많다면 단 몇 %만 떼어 놓아도 전체 데이터를 대표하는데 문제가 없다

# 훈련 세트에서 모델을 훈련하고 검증 세트로 모델을 평가한다
# 이런 식으로 테스트하고 싶은 매개변수를 바꿔가며 가장 좋은 모델을 고른다
# 그 다음 이 매개변수를 사용해 훈련 세트와 검증 세트를 합쳐 전체 훈련 데이터에서 모델을 다시 훈련한다
# 그리고 마지막에 테스트 세트에서 최종 점수를 평가한다

# 판다스로 csv 데이터 읽기
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

# class 열을 타깃으로 사용하고 나머지 열은 특성 배열에 저장
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련 세트와 테스트 세트를 나눈다
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# train_input과 train_target을 다시 train_test_split() 함수에 넣어 훈련 세트 sub_input, sub_target과 
# 검증 세트 val_input, val_target을 만든다
# 여기에서도 test_size 매개변수를 0.2로 지정하여 train_input의 약 20%를 val_input으로 만든다
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

# 훈련 세트와 검증 세트의 크기 확인
print(sub_input.shape, val_input.shape)
# (4157, 3) (1040, 3)

# 모델 생성 후 평가
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
# 0.9971133028626413
print(dt.score(val_input, val_target))
# 0.864423076923077

# 훈련 세트에 과대적합

# 교차 검증
# 검증 세트를 만드느라 훈련 세트가 줄어들었다
# 보통 많은 데이터를 훈련에 사용할수록 좋은 모델이 만들어진다
# 그렇다고 검증 세트를 너무 조금 떼어 놓으면 검증 점수가 들쭉날쭉하고 불안정할 것이다
# 이럴때 교차 검증(cross validation)을 이용하면 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터를 사용할 수 있다

# 교차 검증은 검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복한다
# 그다음 이 점수를 평균하여 최종 검증 점수를 얻는다

# 3-폴드 교차 검증
# 훈련 세트를 세 부분으로 나눠서 교차 검증을 수행하는 것을 3-폴드 교차 검증이라고 한다
# 통칭 k-폴드 교차 검증(k-fold cross validation)이라고 하며 훈련 세트르 ㄹ몇 부분으로 나누냐에 따라 다르게 부른다
# k-겹 교차 검증이라고도 부른다

# 보통 5-폴드 교차 검증이나 10-폴드 교차 검증을 많이 사용한다 
# 이렇게 하면 데이터의 80~90%까지 훈련에 사용할 수 있다
# 검증 세트가 줄어들지만 각 폴드에서 계산한 검증 점수를 평균하기 때문에 안정된 점수로 생각할 수 있다

# 사이킷런에는 cross_validate() 라는 교차 검증 함수가 있다
# 사용법은 간단한데 먼저 평가할 모델 객체를 첫 번째 매개변수로 전달한다
# 그 다음 앞에서처럼 직접 검증 세트를 떼어 내지 않고 훈련 세트 전체를 cross_validate() 함수에 전달

# 사이킷런에는 cross_validate() 함수의 전신인 cross_val_score() 함수도 있다
# 이 함수는 cross_validate() 함수의 결과 중에서 test_score 값만 반환
from sklearn.model_selection import cross_validate

scores = cross_validate(dt, train_input, train_target)
print(scores)

# {'fit_time': array([0.03056216, 0.01651955, 0.01126099, 0.01042104, 0.01005459]), 
# 'score_time': array([0.00607395, 0.00175476, 0.00167656, 0.00151825, 0.00156927]), 
# 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}

# 교차 검증의 최종 점수는 test_score 키에 담긴 5개의 점수를 평균하여 얻을 수 있다
# 이름은 test_score지만 검증 폴드의 점수이다 
import numpy as np

print(np.mean(scores['test_score']))
# 0.855300214703487

# 교차 검증을 수행하면 입력한 모델에서 얻을 수 있는 최상의 검증 점수를 가늠해 볼 수 있다.

# 한 가지 주의할 점은 cross_validate()는 훈련 세트를 섞어 폴드를 나누지 않는다
# 앞서 train_test_split() 함수로 전체 데이터를 섞은 후 훈련 세트를 준비했기 때문에 따로 섞을 필요가 없다
# 만약 교차 검증을 할 때 훈련 세트를 섞으려면 분할기(splitter)를 지정해야 한다

# 사이킷런의 분할기는 교차 검증에서 폴드를 어떻게 나눌지 결정해 준다
# cross_validate() 함수는 기본적으로 회귀 모델일 경우 KFold 분할기를 사용하고 분류 모델일 경우 타깃 클래스를 고루 나누기 위해
# StraifiedKFold를 사용한다

# 즉 앞서 수행한 교차 검증은 다음 코드와 동일
from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
# 0.855300214703487

# 만약 훈련 세트를 섞은 후 10-폴드 교차 검증을 수행하려면 다음과 같이 작성해야 한다
# n_splits 매개변수는 몇(k) 폴드 교차 검증을 할지 정한다
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
# 0.8574181117533719

# KFold 클래스도 동일한 방식으로 사용할 수 있다

# 하이퍼 파라미터 튜닝
# 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터를 하이퍼파라미터라고 한다
# 사이킷런과 같은 머신러닝 라이브러리를 사용할 때 이런 하이퍼파라미터는 모두 클래스나 메서드의 매개변수로 표현된다

# 하이퍼파라미터 튜닝 작업
# 라이브러리가 제공하는 기본값을 그대로 사용해 모델을 훈련 
# 그다음 검증 세트의 점수나 교차 검증을 통해서 매개변수를 조금씩 바꿔 본다
# 모델마다 적게는 1~2개에서 많게는 5~6개의 매개변수를 제공한다

# 사람의 개입 없이 하이퍼파라미터 튜닝을 자동으로 수행하는 기술을 AutoML이라고 부른다

# 예를 들어
# 가령 결정 트리 모델에서 최적의 max_depth 값을 찾았다고 가정할 때
# 그다음 max_depth를 최적의 값으로 고정하고 min_samples_split을 바꿔가며 최적의 값을 찾는다
# 이렇게 한 매개변수의 최적값을 찾고 다른 매개변수의 최적값을 찾으면 안된다
# max_depth의 최적값은 min_sample_split 매개변수의 값이 바뀌면 함께 달라진다
# 즉 이 두 매개변수를 동시에 바꿔가며 최적의 값을 찾아야 한다

# 게다가 매개변수가 많아지면 문제는 더 복잡해진다
# 파이썬의 for 반복문으로 이런 과정을 직접 구현할 수도 있지만
# 사이킷런에서 제공하는 그리드 서치(Grid Search)를 사용한다

# 사이킷런의 GridSearchCV 클래스는 친절하게도 하이퍼파라미터 탐색과 교차 검증을 한 번에 수행한다
# 별도로 cross_validate() 함수를 호출할 필요가 없다

# 어떻게 사용하는지 간단한 예
# 기본 매개변수를 사용한 결정 트리 모델에서 min_impurity_decrease 매개변수의 최적값 찾기
# 먼저 GridSearchCV 클래스를 임포트하고 탐색할 매개변수와 탐색할 값의 리스트를 딕셔너리로 만든다
from sklearn.model_selection import GridSearchCV

# 0.0001 부터 0.0005까지 0.0001씩 증가하는 5개의 값을 시도
prams = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
# 결정 트리 클래스의 객체를 생성하자마자 바로 전달
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), prams, n_jobs= -1)

# 일반 모델을 훈련하는 것처럼 gs 객체에 fit() 메서드를 호출한다
# 이 메서드를 호출하면 그리드 서치 객체는 결정 트리 모델 min_impurity_decrease 값을 바꿔가며 총 5번 실행한다

# GridSearchCV의 cv 매개변수 기본값은 5이다
# 따라서 min_impurity_decrease 값마다 5-폴드 교차 검증을 수행한다
# 결국 5 * 5 = 25개의 모델을 훈련한다
# 많은 모델을 훈련하기 때문에 GridSearchCV 클래스의 n_jobs 매개변수에서 병렬 실행에 사용할 CPU 코어 수를 지정하는 것이 좋다
# 이 매개변수의 기본값은 1이다 
# -1로 지정하면 시스템에 있는 모든 코어를 사용한다

gs.fit(train_input, train_target)

# 교차 검증에서 최적의 하이퍼파라미터를 찾으면 전체 훈련 세트로 모델을 다시 만들어야 하지만 
# 사이킷런의 그리드 서치는 훈련이 끝나면 25개의 모델 중에서 검증 점수가 가장 높은 모델의 매개변수 조합으로
# 전체 훈련 세트에서 자동으로 다시 모델을 훈련한다

# 이 모델은 gs 객체의 best_estimator_ 속성에 저장되어 있다
# 이 모델을 일반 결정 트리처럼 똑같이 사용할 수 있다
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
# 0.9615162593804117

# 그리드 서치로 찾은 최적의 매개변수는 best_params_ 속성에 저장되어 있다
print(gs.best_params_)
# {'min_impurity_decrease': 0.0001}

# 각 매개변수에서 수행한 교차 검증의 평균 점수는 cv_results_ 속성의 mean_test_score키에 저장되어 있다
# 5번의 교차 검증으로 얻은 점수를 출력
print(gs.cv_results_['mean_test_score'])
# [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]

# 가장 큰 값의 인덱스를 넘파이 argmax()함수로 추출하고 이 인덱스를 사용해 prams 키에 저장된 매개변수를 출력할 수 있다
# 이 값이 최상의 검증 점수를 만든 매개변수 조합이다
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
# {'min_impurity_decrease': 0.0001}

# 이 과정을 정리하면 다음과 같다
# 1. 탐색할 매개변수를 지정
# 2. 훈련 세트에서 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합을 찾는다
# 이 조합은 그리드 서치 객체에 저장
# 3. 그리드 서치는 최상의 매개변수에서 (교차 검증에 사용한 훈련 세트가 아니라) 전체 훈련 세트를 
# 사용해 최종 모델을 훈련. 이 모델도 그리드 서치 객체에 저장

# 결정 트리에서 min_impurity_decrease는 노드를 분할하기 위한 불순도 감소 최소량을 지정
# 여기에다가 max_depth로 트리의 깊이를 제한하고 min_sample_split으로 노드를 나누기 위한 최소 샘플 수도 추출
params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001), #넘파이 arange() 함수는
# 첫 번째 매개변수에서 두 번째 매개변수에 도달할때까지 세 번째 매개변수를 계속 더한 배열 생성
          'max_depth': range(5, 20, 1), #파이썬 range는 정수만 사용 가능
          'min_samples_split': range(2, 100, 10)
        }

# 따라서 이 매개변수로 수행할 교차 검증 횟수는 9 * 15 * 10 = 1350개이다
# 기본 5폴드 교차검증을 수행하므로 만들어지는 모델의 수는 6750개다 
# cpu를 전부 사용하기 위해 n_jobs를 -1로 지정해서 그리드 서치 진행
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs= -1)
gs.fit(train_input, train_target)

# 최상의 매개변수 조합 확인
print(gs.best_params_)
# {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}

# 최상의 교차 검증 점수 확인
print(np.max(gs.cv_results_['mean_test_score']))

# 랜덤 서치
# 매개변수의 값이 수치일 때 값의 범위나 간격을 미리 정하기 어려울 수 있다
# 너무 많은 매개 변수 조건이 있어 그리드 서치 수행 시간이 오래 걸릴 수 있다
# 이럴 때 랜덤 서치(Random Search)를 사용하면 좋다

# 핸덤 서치에는 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링할 수 있는 확률 분포
# 객체를 전달한다

# 싸이파이(scipy)
# 파이썬의 핵심 과학 라이브러리 중 하나
# 적분, 보간, 선형 대수, 확률 등을 포함한 수치 계산 전용 라이브러리
# 사이킷런은 넘파이와 싸이파이 기능을 많이 사용한다
from scipy.stats import uniform, randint

# 싸이파이의 stats 서브 패키지에 있는 uniform과 randint 클래스는 모두 주어진 범위에서 고르게 값을 뽑는다
# 이를 균등 분포에서 샘플링한다 라고 말한다
# randint는 정숫값을 뽑고 uniform은 실숫값을 뽑는다
# 사용법은 같다

# 10개의 숫자 샘플링
rgen = randint(0,10)
rgen.rvs(10)
# array([0, 9, 2, 6, 3, 4, 0, 1, 3, 5])

# 1000개를 샘플링해서 각 숫자의 개수를 출력
np.unique(rgen.rvs(1000), return_counts=True)
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
#  array([102, 101,  84,  95, 102,  90,  97, 100, 116, 113]))
# 숫자가 어느 정도 고르게 추출된다

# uniform 클래스로 0~1 사이의 10개 실수 추출
ugen = uniform(0, 1)
ugen.rvs(10)
# array([0.26597489, 0.1488865 , 0.72663052, 0.67215681, 0.9631752 ,
#        0.69133256, 0.04281637, 0.60542112, 0.62241722, 0.90843272])

# 랜덤 서치에 randint과 uniform 클래스 객체를 넘겨주고 총 몇 번을 샘플링해서 최적의 매개변수를 찾으라고 명령할 수 있다
# 샘플링 횟수는 시스템 자원이 허락하는 범위 내에서 최대한 크게 하는 것이 좋다

# 탐색할 매개변수의 딕셔너리를 만드는데 여기에서 min_samples_leaf 매개변수를 탐색 대상에 추가
# 이 매개변수는 리프 노드가 되기 위한 최소 샘플의 개수이다
# 어떤 노드가 분할하여 만들어질 자식 노드의 샘플 수가 이 값보다 작을 경우 분할하지 않는다
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
}

# 샘플링 횟수는 사이킷런의 랜덤 서치 클래스인 RandomizedSearchCV의 n_iter 매개변수에 지정
from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

# 총 100번(n_iter 매개변수)을 샘플링하여 교차 검증을 수행하고 최적의 매개변수 조합을 찾는다
# 앞서 그리드 서치보다 훨씬 교차 검증 수를 줄이면서 넓은 영역을 효과적으로 탐색할 수 있다

# 최적의 매개변수 조합 출력
print(gs.best_params_)
# {'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}

# 교차검증 점수 출력
print(np.max(gs.cv_results_['mean_test_score']))
# 0.8695428296438884

# 최적의 모델을 최종 모델로 결정하고 테스트 세트의 성능 확인
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
# 0.86

# 테스트 세트 점수는 검증 세트에 대한 점수보다 조금 작은 것이 일반적이다