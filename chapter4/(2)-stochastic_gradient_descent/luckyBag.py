# 점진적 학습
# 온라인 학습이라고도 부른다
# 훈련한 모델을 버리지 않고 새로운 데이터에 대해서만 조금씩 더 훈련하는 방식
# 훈련에 사용한 데이터를 모두 유지할 필요도 없고 앞서 학습한 내용이 사라지지도 않는다

# 확률적 경사 하강법(Stochastic gradient descent)
# 대표적인 점진적 학습 알고리즘 
# 확률적 경사 하강법에서 확률적이란 말은 무작위하게 혹은 랜덤하게의 기술적인 표현
# 경사 하강법은 경사를 따라 내려가는 방법을 말한다

# 경사 하강법
# 가장 가파른 경사를 따라 원하는 지점에 도달하는 것이 목표
# 가장 가파른 길을 찾아 내려오지만 조금씩 내려오는 것이 중요하다
# 이렇게 내려오는 과정이 경사 하강법 모델을 훈련하는 것이다

# 확률적
# 경사 하강법은 훈련 세트를 사용하여 가장 가파른 길을 찾는다
# 전체 샘플을 사용하지 않고 딱 하나의 샘플을 훈련 세트에서 랜덤하게 골라 가장 가파른 길을 찾는다

# 확률적 경사 하강법은 훈련 세트에서 랜덤하게 하나의 샘플을 선택하여 가파른 경사를 조금 내려간다
# 그다음 훈련 세트에서 랜덤하게 또 다른 샘플을 하나 선택하여 경사를 조금 내려간다
# 이런 식으로 전체 샘플을 모두 사용할 때까지 계속한다
# 모든 샘플을 다 사용해도 경사의 끝에 도달하지 못했으면 다시 처음부터 시작한다

# 에포크(epoch)
# 확률적 경사 하강법에서 훈련 세트를 한 번 모두 사용하는 과정을 에포크라고 부른다
# 일반적으로 경사 하강법은 수십, 수백 번 이상 에포크를 수행한다

# 미니배치 경사 하강법(minibatch gradient descent)
# 1개의 샘플이 아니라 무작위로 몇 개의 샘플을 선택해서 경사를 내려가는 방식
# 실전에서 아주 많이 사용한다

# 배치 경사 하강법(batch gradient descent)
# 극단적으로 한 번 경사로를 따라 이동하기 위해 전체 샘플을 사용하는 방식
# 전체 데이터를 사용하기 때문에 가장 안정적인 방법이지만 그만큼 컴퓨터 자원을 많이 사용한다
# 어떤 경우는 데이터가 너무 많아 한 번에 전체 데이털르 모두 읽을 수 없을수도 있다

# 이 때문에 훈련 데이터가 모두 준비되어 있지 않고 매일매일 업데이트 되어도 학습을 계속 이어나갈 수 있다

# 신경망 알고리즘은 학률적 경사 하강법을 반드시 사용하는 알고리즘
# 신경망은 일반적으로 많은 데이터를 사용하기 때문에 한 번에 모든 데이터를 사용하기 어렵다
# 또 모델이 매우 복잡하기 때문에 수학적인 방법으로 해답을 얻기어렵다
# 신경망 모델은 확률적 경사 하강법이나 미니배치 경사 하강법을 사용한다

# 손실 함수(loss function)
# 어떤 문제에서 머신러닝 알고리즘이 얼마나 엉터리인지를 측정하는 기준
# 손실 함수의 값이 작을수록 좋다
# 하지만 어떤 값이 최솟값인지는 알지 못한다
# 가능한 많이 찾아보고 만족할만한 수준이라면 다 내려왔다고 인정해야 한다

# 비용 함수(cost function)는 손실 함수의 다른 말이다
# 엄밀히 말하면 손실 함수는 샘플 하나에 대한 손실을 정의하고 비용 함수는 훈련 세트에 있는 모든 샘플에 대한 손실 함수의 합을 말한다
# 보통 이 둘을 엄격히 구분하지 않고 섞어서 사용한다

# 분류에서의 손실은 정답을 맞추지 못하는 것이다
# 도미와 빙어를 구분하는 이진 분류 문제를 예시로
# 도미 = 양성 클래스(1) / 빙어 = 음성 클래스(0)
# 예측   정답
#  1   =  1
#  0  !=  1
#  0   =  0
#  1  !=  0

# 4개의 예측 중에 2개만 맞았으므로 정확도는 1/2 = 0.5이다
# 정확도에 음수를 취하면 -1.0이 가장 낮고 -0.0이 가장 높다
# 정확도에는 치명적인 단점이 있는데 앞의 예시와 같이 4개의 샘플만 있다면 가능한 정확도는 0, 0.25, 0.5, 0.75, 1 다섯 가지 뿐이다
# 정확도가 듬성듬성하다면 경사 하강법을 이용해 조금씩 움직일 수 없다
# 경사면은 연속적이어야 한다
# 기술적으로 말하면 손실 함수는 미분 가능해야 한다

# 로지스틱 손실 함수
# 4개의 예측 활률을 각각 0.9, 0.3, 0.2, 0.8이라고 가정

# 첫 번째
# 첫 번째 샘플의 에측은 0.9 이므로 양성 클래스의 타깃인 1과 곱한 다음 음수로 바꿀 수 있다
# 이 경우 예측이 1에 가까울수록 좋은 모델이다
# 예측이 1에 가까울수록 예측과 타깃의 곱의 음수는 점점 작아진다
# 이 값을 손실 함수로 사용한다
# 0.9(예측) * 1(타깃) = 0.9 -> -0.9

# 두 번째
# 두 번째도 마찬가지로 계산
# 0.3(예측) * 1(타깃) = 0.3 -> -0.3

# 세 번째
# 해당 샘플의 타깃은 음성 클래스라 0이다 
# 이 값을 계산하면 무조건 0이 되므로 타깃을 마치 양성 클래스처럼 바꾸어 1로 만든다
# 대신 예측값도 양성 클래스에 대한 예측으로 바꾼다
# 즉 1 - 0.2 = 0.8 로 사용한 후 
# 0.8(예측) * 1(타깃) = 0.8 -> -0.8

# 네 번째
# 샘플의 타깃이 음성 클래스
# 세 번째 샘플과 같은 방식으로 계산한다

# 여기에서 예측 확률에 로그 함수를 적용하면 더 좋다
# 예측 확률의 범위는 0~1 사이인데 로그 함수는 이 사이에서 음수가 되므로 최종 손실 값은 양수가 된다
# 손실이 양수가 되면 이해하기 더 쉽다
# 또 로그 함수는 0에 가까울수록 아주 큰 음수가 되기 때문에 손실을 아주 크게 만들어 모델에 큰 영향을 미칠 수 있다

# 양성 클래스 (타깃 = 1)일 때 손실은 -log(예측 확률)로 계산
# 확률이 1에서 멀어져 0에 가까워질수록 손실이 아주 큰 양수가 된다

# 음성 클래스 (타깃 = 0)일 때 손실은 -log(1-예측 확률)로 계산 
# 예측 확률이 0에서 멀어져 1에 가까워질수록 손실은 아주 큰 양수가 된다

# 이 손실 함수를 로지스틱 손실 함수(logistic loss function) 또는 이진 크로스엔트로피 손실 함수(binary cross-entropy logg function)
# 이라고 부른다
# 이 손실 함수를 사용하면 로지스틱 회귀 모델이 만들어진다

# 다중 분류에서 사용하는 손실 함수를 크로스엔트로피 손실 함수(cross-entropy loss function)라고 부른다
# 문제에 잘 맞는 손실 함수가 개발되어 있어 손실 함수를 우리가 직접 만드는 일은 거의 없다

# 회귀에서는 평균 절댓값 오차를 손실 함수로 사용할 수 있다
# 또는 평균 제곱 오차(mean squared error)를 많이 사용한다 
# 타깃에서 예측을 뺀 값을 제곱한 다음 모든 샘플에 평균한 값이다 
# 이 값이 작을수록 좋은 모델이다

# 머신러닝 라이브러리를 활용한 손실 함수 계산
# 판다스 데이터프레임 생성
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')

# Species열을 제외한 5개의 데이터를 입력 데이터로 가공 
# Species 열은 타깃 데이터로 사용
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# train_test_split() 함수로 훈련 세트와 테스트 세트 생성
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 훈련 세트와 테스트 세트의 특성을 표준화 전처리
# 훈련 세트에서 학습한 통계 값으로 테스트 세트도 변환해야 한다
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transfrom(train_input)
test_scaled = ss.transfrom(test_input)

# 확률적 경사 하강법을 제공하는 대표적인 분류용 클래스는 SGDClassifier이다
from sklearn.linear_model import SGDClassifier

# SGDClassifier 객체를 생성 시 2개의 매개변수를 지정한다
# loss는 손실 함수의 종류를 기정한다 여기서는 loss='log_loss'를 입력하여 로지스틱 손실 함수 지정
# max_iter는 수행할 에포크 횟수를 지정 여기서는 10으로 지정하여 전체 훈련 세트를 10회 반복

# 다중 분류일 경우 SGDClassifier에 loss='log_loss'로 지정하면 클래스마다 이진 분류 모델을 만든다
# 즉 도미는 양성 클래스로 두고 나머지를 모두 음성 클래스로 두는 방식 이런 방식을 OvR(One versus Rest)이라고 부른다

sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
# 0.773109243697479
print(sc.score(test_scaled, test_target))
# 0.775

# 확률적 경사 하강법은 점진적 합습이 가능하다
# SGDClassifier 객체를 다시 만들지 않고 훈련한 모델 sc를 추가로 더 훈련
# 모델을 이어서 훈련할 때는 partial_fit() 메서드 사용
# fit() 메서드와 사용법이 같지만 호출할 때마다 1에포크씩 이어서 훈련 가능

sc.partial_fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
# 0.8151260504201681
print(sc.score(test_scaled, test_target))
# 0.85

# SGDClassifier는 미니배치 경사 하강법이나 배치 하강법을 제공하지 않는다

# 확률적 경사 하강법을 사용한 모델은 에포크 횟수에 따라 과소적합이나 과대적합이 될 수 있다

# 에포크 횟수가 적으면 모델이 훈련 세트를 덜 학습한다
# 마치 산을 다 내려오지 못하고 훈련을 마치는 셈이다 
# 에포크 횟수가 충분히 많으면 훈련 세트를 완전히 학습할 것이다 
# 훈련 세트에 아주 잘 맞는 모델이 만들어진다

# 적은 에포크 횟수로 훈련한 모델은 훈련 세트와 테스트 세트에 잘 맞지 않는 과소적합된 모델일 가능성이 높다
# 반대로 많은 에포크 횟수 동안에 훈련한 모델은 훈련 세트에 너무 잘 맞아 테스트 세트에는 오히려 점수가 나쁜 과대적합된 모델일 가능성이 높다

# 훈련 세트 점수는 에포크가 진행될수록 꾸준히 증가하지만 테스트 세트 점수는 어느 순간 감소하기 시작한다
# 이 구간이 모델이 과대적합되기 시작하는 곳이다

# 과대적합이 시간하기 전에 훈련을 멈추는 것을 조기 종료(early sopping)이라고 한다 

# partial_fit() 메서드를 사용하여 에포크 그래프 만들기
import numpy as np

sc = SGDClassifier(loss='log_loss', random_state=42)

train_score = []
test_score = []

# 종을 중복을 제거하여 추출
classes = np.unique(train_target)

# 300번의 에포크 동안 훈련 반복
# 에포그 반복 횟수에 대한 점수를 출력하기 위해 기존 훈련 정보를 남겨놓는 partial_fit으로 훈련
# _는 버리는 변수 여기서는 반복 횟수를 임시 저장하기 위한 용도로 사용
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)

    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

# 300번의 에포크 반복 후 점수를 그래프로 변환
import matplotlib.pyplot as plt

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 그래프를 출력해보면 백 번째 에포크 이후에는 훈련 세트와 테스트 세트의 점수가 조금씩 벌어진다
# 에포크 초기에는 과소적합되어 훈련 세트와 테스트 세트의 점수가 낮다
# 이 모델의 적정 에포크 횟수는 100회로 보인다

# SGDClassifier는 일정 에포크 동안 성능이 향상되지 않으면 더 훈련하지 않고 자동으로 멈춘ㄴ다
# tol 매개변수에서 향상될 최솟값을 지정 None으로 지정하면 자동으로 멈추지 않는다
sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_input, train_target)

print(sc.score(train_scaled, train_target))
# 0.957983193277311
print(sc.score(test_scaled, test_target))
# 0.925

# 회귀 모델에서 확률적 경사 하강법을 사용한 모델 = SGDRegressor

# SGDClassifier의 loss 매개변수 기본값은 hinge다 
# 힌지 손실(hinge loss)은 서포트 벡터 머신(support vector machine)이라 불리는 또 다른 머신러닝 알고리즘을 위한 손실 함수이다
# 서포트 벡터 머신은 널리 사용하는 머신러닝 알고리즘 중 하나이다

# 힌지 손실을 사용해 훈련 예시
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print(sc.score(train_scaled, train_target))
# 0.9495798319327731
print(sc.score(test_scaled, test_target))
# 0.925