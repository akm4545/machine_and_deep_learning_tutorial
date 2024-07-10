인공 신경망
생물학적 뉴런에서 영감을 받아 만든 머신러닝 알고리즘
이름이 신경망이지만 실제 우리 뇌를 모델링한 것은 아니다
신경망은 기존의 머신러닝 알고리즘으로 다루기 어려웠던 이미지,음성,텍스트 분야에서 
뛰어난 성능을 발휘하면서 크게 주목받고 있다
인공 신경망 알고리즘을 종종 딥러닝이라고도 부른다

텐서플로
구글이 만든 딥러닝 라이브러리로 매우 인기가 높다
CPU와 GPU를 사용해 인공 신경망 모델을 효율적으로 훈련하며 모델 구축과
서비스에 필요한 다양한 도구를 제공
텐서플로 2.0부터는 신경망 모델을 빠르게 구성할 수 있는 케라스를 핵심 API로 채택
케라스를 사용하면 간단한 모델에서 아주 복잡한 모델까지 손쉽게 만들 수 있다

밀집층
가장 간단한 인공 신경망의 층 인공 신경망에는 여러 종류의 층이 있다
밀집층에서는 뉴런들이 모두 연결되어 있기 때문에 완전 연결 층이라고도 부른다
특별히 출력층에 밀집층을 사용할 때는 분류하려는 클래스와 동일한 개수의 뉴런을 사용

원-핫 인코딩
정숫값을 배열에서 해당 정수 위치의 원소만 1이고 나머지는 모두 0으로 변환
이런 변환이 필요한 이유는 다중 분류에서 출력층에서 만든 확률과 크로스 엔트로피 손실을 계산하기 위해서이다
텐서플로에서는 sparse_categorical_entropy 손실을 지정하면 이런 변환을 수행할 필요가 없다

TensorFlow
1. Dense
신경망에서 가장 기본 층인 밀집층을 만드는 클래스
이 층에 첫 번째 매개변수에는 뉴런의 개수를 지정한다
activation 매개변수에는 사용할 활성화 함수를 지정 대표적으로 sigmoid, softmax 함수가 있다
아무것도 지정하지 않으면 활성화 함수를 사용하지 않는다
케라스의 Sequential 클래스에 맨 처음 추가되는 층에는 input_shape 매개변수로 입력의 크기를 지정

2. Sequential
케라스에서 신경망 모델을 만드는 클래스
이 클래스의 객체를 생성할 때 신경망 모델에 추가할 층을 지정할 수 있다
추가할 층이 1개 이상일 경우 파이썬 리스트로 전달

3. compile()
모델 객체를 만든 후 훈련하기 전에 사용할 손실 함수와 측정 지표 등을 지정하는 메서드
loss 매개변수에 손실 함수를 지정
이진 분류일 경우 binary_crossentropy
다중 분류일 경우 categorical_crossentropy
클래스 레이블이 정수일 경우 sparse_categorical_crossetropy로 지정
회귀 모델일 경우 mean_square_error등으로 지정할 수 있다
metrics 매개변수에 훈련 과정에서 측정하고 싶은 지표를 지정할 수 있다
측정 지표가 1개 이상일 경우 리스트로 전달

4. fit()
모델을 훈련하는 메서드
첫 번째와 두 번째 매개변수에 입력과 타깃 데이터를 전달
epochs 매개변수에 전체 데이터에 대해 반복할 에포크 횟수를 지정

5. evaluate()
모델 성능을 평가하는 메서드
첫 번째와 두 번째 매개변수에 입력과 타깃 데이터를 전달
compile() 메서드에서 loss 매개변수에 지정한 손실 함수의 값과 metrics 매개변수에서 지정한 측정 지표를 출력





사이킷런 모델 사용법
모델                       손실함수    반복 횟수
sc   = SGDClassifier(loss='log_loss', max_iter=5)
훈련
sc.fit(train_scaled, train_target)
평가
sc.score(val_scaled, val_target)

케라스 모델
                     층 생성
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
모델
model = keras.Sequential(dense)
                    손실 함수
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
훈련                                  반복 횟수
model.fit(train_scaled, train_target, epochs=5)
평가
model.evaluate(val_scaled, val_target)