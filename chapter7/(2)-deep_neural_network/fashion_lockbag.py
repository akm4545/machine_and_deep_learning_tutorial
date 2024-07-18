# 케라스 API를 사용해서 패션 MNIST 데이터셋 불러오기
from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split

# 이미지 픽셀값을 0~255 범위에서 0~1 사이로 변환
train_scaled = train_input / 255.0
# 28 * 28 크기의 2차원 배열을 784 크기의 1차원 배열로 펼치기
train_scaled = train_scaled.reshape(-1, 28 * 28)
# 훈련, 검증세트 분할
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 은닉충(hidden layer)
# 입력층과 출력층 사이에 밀집층이 추가된 것
# 입력층과 출력층 사이에 있는 모든 층을 은닉층이라고 부른다

# 활성화 함수
# 신경망 층의 선형 방정식의 계산 값에 적용하는 함수
# 이전 학습에서 출력층에 적용했던 소프트맥스 함수도 활성화 함수이다
# 출력층에 적용하는 활성화 함수는 종류가 제한되어 있다
# 이진 분류일 경우 시그모이드 함수를 사용하고 다중 분류일 경우 소프트맥스 함수는 사용한다
# 이에 비해 은닉층의 활성화 함수는 비교적 자유롭다 
# 대표적으로 시그모이드 함수와 볼 렐루(ReLU) 함수 들을 사용한다

# 회귀의 출력은 임의의 어떤 숫자이므로 활성화 함수를 적용할 필요가 없다
# 즉 출력층의 선형 방정식의 계산을 그대로 출력한다
# 이렇게 하려면 Dense 층의 activation 매개변수에 아무런 값을 지정하지 않는다

# 2개의 선형 방정식
# a * 4 + 2 = b -> b * 3 - 5 = c
# 첫 번째 식에서 계산된 b가 두 번째 식 c를 계산하기 위해 쓰임
# 두 번째 식에 첫 번째 식을 대입하면 하나로 합쳐짐
# a * 12 + 1 = c 
# 이렇게 되면 b는 사라지고 b가 하는 일이 없는 셈이다

# 신경망도 마찬가지로 은닉층에서 선형적인 산술 계산만 수행한다면 수행 역할이 없느 ㄴ셈이다
# 선형 계산을 적당하게 비선형적으로 비틀어 주어야 한다
# 그래야 다음 층의 계산과 단순히 합쳐지지 않고 나름의 역할을 할 수 있다
# 마치 다음과 같다
# a * 4 + 2 = b -> log(b) = k -> k * 3 - 5 = c

# 인공 신경망을 그림으로 나타낼 때 활성화 함수를 생략하는 경우가 많은데 이는 절편과 마찬가지로
# 번거로움을 피하기 위해서 활성화 함수를 별개의 층으로 생각하지 않고 층에 포함되어 있다고 간주하기 때문
# 모든 신경망의 은닉층에는 항상 활성화 함수가 있다

# 많이 사용하는 활성화 함수는 시그모이드 함수이다
# 이 함수는 뉴런의 출력값을 0과 1 사이로 압축한다

# 시그모이드 활성화 함수를 사용한 은닉층과 소프트맥스 함수를 사용한 출력층을
# 케라스의 Dense 클래스로 생성
# 케라스에서 신경망의 첫 번째 층은 input_shape 매개변수로 입력의 크기를 꼭 지정해 주어야 한다

# 은닉층
# 100개의 뉴런을 가진 밀집층 
# 활성화 함수를 sigmoid로 지정하고 input_shape 매개변수에서 입력의 크기를 (784,)로 지정
dense1 = keras.layers.Dense(100, activation='sigmoid', input_sahpe=(784,))
# 출력층
# 10개의 클래스를 분류하므로 10개의 뉴런을 두었다
# 활성화 함수 소프트맥스
dense2 = keras.layers.Dense(10, activation='softmax')

# 은닉층의 뉴런 개수를 정하는데는 특별한 기준이 없다
# 몇 개의 뉴런을 두어야 할지 판단하기 위해서는 상당한 경험이 필요하다
# 한 가지 제약 사항이 있다면 적어도 출력층의 뉴런보다는 많게 만들어야 한다
# 클래스 10개에 대한 확률을 예측해야 하는데 이전 은닉층의 뉴런이 10개보다 적다면 부족한 정보가 전달될 것이다

# 심층 신경망(deep neural network, DNN)
model = keras.Sequential([dense1, dense2])

# 여러 개의 층을 추가하려면 리스트로 만들어서 전달
# 출력층을 가장 마지막에 두어야 한다
# 이 리스트는 가장 처음 등장하는 은닉층에서 마지막 출력층의 순서로 나열

# 인공 신경망의 강력한 성능은 층을 추가하여 입력 데이터에 대해 연속적인 학습을 
# 진행하는 능력에서 나온다

# 케라스는 모델의 summary() 메서드를 호출하면 층에 대한 정보를 얻을 수 있다
model.summary()
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  dense (Dense)               (None, 100)               78500     
                                                                 
#  dense_1 (Dense)             (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 79510 (310.59 KB)
# Trainable params: 79510 (310.59 KB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# 층이 순서대로 나열
# 층 이름, 클래스, 출력 크기, 모델 파라미터 개수 출력
# 층을 만들 때 name 매개변수로 이름 지정 가능 / 지정하지 않으면 케라스가 자동으로 dense라고 이름붙임

# 출력 크기가 (None, 100)으로 출력된다
# 첫 번째 차원은 샘플의 개수를 나타낸다
# 샘플 개수가 아직 정의되어 있지 않기 때문에 None이다

# 케라스 모델의 fit() 메서드에 훈련 데이터를 주입하면 이 데이터를 한 번에 모두 사용하지 않고
# 잘게 나누어 여러 번에 걸쳐 경사 하강법 단계를 수행한다 (미니배치 경사 하강법)
# 케라스의 기본 미니배치 크기는 32개다
# 이 값은 fit() 메서드에서 batch_size 매개변수로 바꿀 수 있다
# 따라서 샘플 개수를 고정하지 않고 어떤 배치 크기에도 유연하게 대응할 수 있도록 None으로 설정
# 신경망 층에 입력되거나 출력되는 배열의 첫 번째 차원을 배치 차원이라고 부른다

# 두 번째 100은 특성의 개수 -> 은닉층의 뉴런 개수를 100개로 두었으니 100개의 출력이 나온다
# 샘플 784개의 픽셀값이 은닉층을 통과하면서 100개의 특성으로 압축

# 마지막은 모델 파라미터 개수 
# 이 층은 Dense 층이므로 입력 픽셀 784개와 100개의 모든 조합에 대한 가중치가 있다
# 그리고 뉴런마다 1개의 절편이 있다
# 784(픽셀) * 100(출력) + 100(절편) = 78500
# 두 번쨰 층의 모델 파라미터 개수
# 100(은닉층 출력 개수) * 10(출력층 출력) + 10(절편) = 1010

# summary 메서드 출력 하단에 총 모델 파리미터 개수, 훈련되는 파라미터 개수가 나온다
# 간혹 경사 하강법으로 훈련되지 않는 파라미터를 가진 층이 있는데 이런 층의 파라미터 개수는
# 훈련되지 않은 파라미터로 나온다

# Dense 클래스의 객체를 따로 저장하여 쓸 일이 없기 때문에 
# Sequential 클래스의 생성자 안에서 바로 Dense 클래스의 객체를 만드는 경우가 많다
model = keras.Sequential([
    keras.layers.Dense(100, activation="sigmoid", input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 MNIST 모델')

# 이렇게 작업하면 추가되는 층을 한눈에 쉽게 알아보는 장점이 있다
# 모델의 이름과 달리 층의 이름은 반드시 영문이여야 한다

model.summary()

# Model: "패션 MNIST 모델"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  hidden (Dense)              (None, 100)               78500     
                                                                 
#  output (Dense)              (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 79510 (310.59 KB)
# Trainable params: 79510 (310.59 KB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# 이 방법이 편리하지만 아주 많은 층을 추가하려면 Sequential 클래스 생성자가 매우 길어진다
# 또 조건에 따라 층을 추가할 수도 없다
# Sequential 클래스에서 층을 추가할 때 가장 널리 사용하는 방법은 모델의 add() 메서드다

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

# 모델 훈련
# 5번의 에포크
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

# Epoch 1/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.5675 - accuracy: 0.8063
# Epoch 2/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.4111 - accuracy: 0.8521
# Epoch 3/5
# 1500/1500 [==============================] - 4s 3ms/step - loss: 0.3760 - accuracy: 0.8646
# Epoch 4/5
# 1500/1500 [==============================] - 4s 3ms/step - loss: 0.3526 - accuracy: 0.8719
# Epoch 5/5
# 1500/1500 [==============================] - 5s 4ms/step - loss: 0.3346 - accuracy: 0.8773

# 추가된 층이 성능을 향샹시켰다

# 초창기 인공 신경망의 은닉층에 많이 사용된 활성화 함수는 시그모이드 함수였다
# 이 함수에는 단점이 있는데 함수의 오른쪽과 왼쪽 끝으로 갈수록 그래프가 누워있기 떄문에
# 올바른 출력을 만드는데 신속하게 대응하지 못한다
# 특히 층이 많은 심층 신경망일수록 그 효과가 누적되어 학습을 더 어렵게 만든다

# 렐루(ReLU)함수
# 이를 개선하기 위해 나온 함수
# 렐루 함수는 아주 간단하다
# 입력이 양수일 경우 마치 활성화 함수가 없는 것처럼 그냥 입력을 동과시키고 음수일 경우 0으로 만든다

# 렐루 함수는 max(0, z)와 같이 쓸 수 있다
# 이 함수는 z가 0보다 크면 z를 출력하고 z가 0보다 작으면 0을 출력한다
# 렐루 함수는 특히 이미지 처리에서 좋은 성능을 낸다고 알려져 있다

# 패션 MNIST 데이터는 28 * 28 크기이기 떄문에 인공 신경망에 주입하기 위해 넘파이 배열의
# reshape() 메서드를 사용해 1차원으로 펼쳤다
# 직접 펼쳐도 좋지만 케라스는 이를 위한 Flatten 층을 제공한다

# Flatten 클래스는 배치 차원을 제외하고 나머지 입력 차원을 모두 일렬로 펼치는 역할만 한다
# 입력에 곱해지는 가중치나 절편이 없다
# 따라서 인공 신경망의 성능을 위해 기여하는 바는 없다
# 하지만 Flatten 클래스를 층처럼 입력층과 은닉층 사이에 추가하기 때문에 이를 층이라 부른다

# Flatten층 추가 
model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

# 이 신경망을 깊이가 3인 신경망이라고 부르지는 않는다
# Flatten 클래스는 학습하는 층이 아니기 떄문이다

model.summary()

# Model: "sequential_6"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten (Flatten)           (None, 784)               0         
                                                                 
#  dense_12 (Dense)            (None, 100)               78500     
                                                                 
#  dense_13 (Dense)            (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 79510 (310.59 KB)
# Trainable params: 79510 (310.59 KB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# Flatten 층이 신경망 모델에 추가되면서 784개의 입력이 첫 번째 은닉층에 전달된다는 
# 것을 알 수 있다는 장점이 있다

# 입력 데이터에 대한 전처리 과정을 가능한 모델에 포함시키는 것이 케라스 API의 철학 중 하나이다

# 훈련 데이터 다시 준비
# reshape() 메서드를 적용하지 않는다
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 모델 컴파일 후 훈련
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)

# Epoch 1/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.5366 - accuracy: 0.8114
# Epoch 2/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.3959 - accuracy: 0.8569
# Epoch 3/5
# 1500/1500 [==============================] - 4s 3ms/step - loss: 0.3557 - accuracy: 0.8708
# Epoch 4/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.3338 - accuracy: 0.8815
# Epoch 5/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.3163 - accuracy: 0.8867

# 시그모이드 함수와 비교하면 성능이 조금 향상되었다

# 검증 세트 성능 확인
model.evaluate(val_scaled, val_target)
# [0.4111086130142212, 0.8600000143051147]

# 은닉층을 추가하지 않은 경우보다 몇 퍼센트 성능이 향상되었다

# 신경망 하이퍼파리미터 종류
# 은닉층 개수
# 은닉층 뉴런 개수
# 활성화 함수
# 층의 종류
# 케라스의 기본 미니배치 경사 하강법의 미니배치 개수 (batch_size)
# fit 메서드의 반복 횟수(epochs)
# compile() 메서드에서는 케라스의 기본 경사 하강법 알고리즘인 RMSprop을 사용했다
# 케라스는 다양한 종류의 경사 하강법 알고리즘을 제공
# 이들을 옵티마이저(optimizer)라고 부른다
# RMSprop의 학습률도 하이퍼 파라미터

# 가장 기본적인 옵티마이저는 확률 경사 하강법인 SGD이다
# 미니배치를 사용

# SGD 옵티마이저 사용 예시
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')

# 이 옵티마이저는 tensorflow.keras.optimizers 패키지 아래 SGD 클래스로 구현
# sgd 문자열은 해당 클래스의 기본 설정 매개변수로 생성한 객체와 동일

# 아래 코드는 위의 코드와 동일하다
sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')

# SGD 클래스의 학습률 기본값이 0.01일 때 이를 바꾸고 싶다면 
# learning_rate 매개변수 지정
sgd = keras.optimizers.SGD(learning_rate=0.1)

# 많이 사용하는 옵티마이저
# 기본 경사 하강법 옵티마이저
# SGD(learning_rate-0.01), 모멘텀(momentum > 0), 네스테로프 모멘텀(nesterov=True)
# 적응적 학습률 옵티마이저
# RMSprop(learning_rate-0.001), Adam(learning_rate-0.001), Adagrad

# 기본 경사 하강법 옵티마이저는 모두 SGD 클래스에서 제공
# SGD 클래스의 momentum 매개변수의 기본값은 0이다 
# 이를 0보다 큰 값으로 지정하면 마치 그레이디언트 가속도처럼 사용하는 모멘텀 최적화(momentum optimization)
# 를 사용
# 보통 momentum 매개변수는 0.9 이상을 지정한다

# SGD 클래스의 nesterov 매개변수를 기본값 False에서 True로 바꾸면 네스테로프 모멘텀 최적화(nesterov momentum optimization)
# [또는 네스테로프 가속 경사]를 사용한다
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)

# 네스테로프 모멘텀은 모멘텀 최적화를 2번 반복하여 구현한다
# 대부분의 경우 네스테로프 모멘텀 최적화가 기본 확률적 경사 하강법보다 더 나은 성능을 제공한다

# 모델이 최적점에 가까이 갈수록 학습률을 낮출 수 있다
# 이렇게 하면 안정적으로 최적점에 수렴할 가능성이 높다
# 이런 학습률을 적응적 학습률(adaptive learning rate)이라고 한다
# 이런 방식들은 학습률 매개변수를 튜닝하는 수고를 덜 수 있는 것이 장점이다

# 적응적 합습률을 사용하는 대표적인 옵티마이저는 Adagrad와 RMSprop이다
# 각각 compile() 메서드의 optimizer 매개변수에 adagrad와 rmsprop으로 지정할 수 있다
# optimizer 매개변수의 기본값은 rmsprop이다 

# 옵티마이저 변경
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')

rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics='accuracy')

# 모멘텀 최적화와 RMSprop의 장점을 접목한 것이 Adam
# Adam은 RMSprop과 함께 맨처음 시도해 볼 수 있는 좋은 알고리즘이다
# keras.optimizers 패키지 아래에 있다
# 적응적 학습률을 사용하는 이 3개의 클래스는 learning_rate 매개변수의 기본값으로 모두 0.001을 사용

# Adam 클래스의 매개변수 기본값을 사용해 패션 MNIST 모델 훈련
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
# Epoch 1/5
# 1500/1500 [==============================] - 6s 3ms/step - loss: 0.5248 - accuracy: 0.8176
# Epoch 2/5
# 1500/1500 [==============================] - 7s 5ms/step - loss: 0.3952 - accuracy: 0.8580
# Epoch 3/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.3536 - accuracy: 0.8714
# Epoch 4/5
# 1500/1500 [==============================] - 8s 5ms/step - loss: 0.3278 - accuracy: 0.8806
# Epoch 5/5
# 1500/1500 [==============================] - 5s 3ms/step - loss: 0.3076 - accuracy: 0.8863
# <keras.src.callbacks.History at 0x7e24ad34a380>

# 기본 RMSprop을 사용했을 때와 거의 같은 성능을 보여준다

# 검증세트 확인
model.evaluate(val_scaled, val_target)
# [0.35007190704345703, 0.8737499713897705]

# 환경마다 차이가 있지만 여기서는 기본 RMSprop보다 조금 나은 성능을 낸다