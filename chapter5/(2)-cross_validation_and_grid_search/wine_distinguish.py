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