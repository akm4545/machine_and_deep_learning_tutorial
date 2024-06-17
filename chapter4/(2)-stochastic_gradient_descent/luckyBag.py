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