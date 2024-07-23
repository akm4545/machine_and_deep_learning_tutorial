# 케라스의 fit() 메서드는 History 클래스 객체를 반환한다
# History 객체에는 훈련 과정에서 계산한 지표, 즉 손실과 정확도 값이 저장되어 있다

# 패션 MNIST 데이터셋을 적재하고 훈련 세트와 검증 세트로 나눔
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 모델을 만드는 간단한 함수 정의
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))

    if a_layer:
        model.add(a_layer)
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# 함수로 모델 생성
model = model_fn()
model.summary()
# Model: "sequential_12"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten_5 (Flatten)         (None, 784)               0         
                                                                 
#  dense_22 (Dense)            (None, 100)               78500     
                                                                 
#  dense_23 (Dense)            (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 79510 (310.59 KB)
# Trainable params: 79510 (310.59 KB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# History 객체 담기
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)

# verbosse 매개변수는 훈련 과정 출력을 조절
# 기본값은 1로 에포크마다 진행 막대와 함께 손실 등의 지표가 풀력
# 2로 바꾸면 진행 막대를 뺴고 출력 
# 0은 훈련 과정을 나타내지 않는다

# history 객체에는 훈련 측정값이 담겨 있는 history 딕셔너리가 들어 있다
print(history.history.keys())
# dict_keys(['loss', 'accuracy'])
# 손실과 정확도가 포함되어있다

# 케라스는 기본적으로 에포크마다 손실을 계산한다
# 정확도는 compile() 메서드에서 metrics 매개변수에 accuracy를 추가했기 떄문에 history 속성에 포함되어 있다
# history 속성에 포함된 손실과 정확도는 에포크마다 계산한 값이 순서대로 나열된 단순 리스트이다

# 맷플롯립으로 그래프 출력
# 손실
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 정확도
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

# 출력시 에포크마다 손실이 감소하고 정확도가 향상된다

# 에포크 횟수 20회로 증가
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0)

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# 손실이 잘 감소한다

# 에포크에 따른 과대적합과 과소적합을 파악하려면 훈련 세트에 대한 점수뿐만 아니라
# 검증 세트에 대한 점수도 필요하다

# 인공 신경망 모델이 최적화하는 대상은 정확도가 아니라 손실 함수이다
# 이따금 손실 감소에 비례하여 정확도가 높아지지 않는 경우도 있다
# 따라서 모델이 잘 훈련되었는지 판단하려면 정확도보다는 손실 함수의 값을 확인하는 것이 더 낫다

# 에포크마다 검증 손실을 계산하기 위해 케라스 모델의 fit() 메서드에 검증 데이터를 전달할 수 있다
# validation_data 매개변수에 검증에 사용할 입력과 타깃값을 듀플로 만들어 전달
model = model_fn()
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

# history 객체의 키 추출
print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
# 검증 세트에 대한 손실은 val_loss
# 정화도는 val_accuracy

# 과대/과소적합 문제를 조사하기 위해 훈련 손실, 검증 손실을 그래프로 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 초기에 검증 손실이 감소하다가 다섯 번째 에포크에서 다시 상승하기 시작한다
# 훈련 손실은 꾸준히 감소하기 때문에 전형적인 과대적합 모델이 만들어진다
# 검증 손실이 상승하는 시점을 가능한 뒤로 늦추면 검증 세트에 대한 손실이 줄어들 뿐만 아니라
# 검증 세트에 대한 정확도도 증가할 것이다

# 옵티마이저 하이퍼파리미터를 조정하여 과대적합 완화
# 기본 RMSprop 옵티마이저는 많은 문제에서 잘 동작한다
# 만약 이 옵티마이저 대신 다른 옵티마이저를 테스트해 본다면 Adam을 사용하자
# Adam은 적응적 학습률을 사용하기 떄문에 에포크가 진행되면서 학습률의 크기를 조정할 수 있다

# Adam 옵티마이저 적용
model = model_fn()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 과대 적합이 훨씬 줄은 그래프가 나온다
# 검증 손실 그래프에 여전히 요동이 남아 있지만 열 번째 에포크까지 전반적인 감소 추세가 이어진다
# 이는 Adam 옵티마이저가 이 데이터셋에 잘 맞는다는것을 보여준다
# 더 나은 손실 곡선을 얻으려면 학습률을 조정해서 다시 시도해 볼 수도 있다

# 드롭아웃(dropout)
# 딥러닝의 아버지로 불리는 제프리 헌틴이 소개
# 이 방식은 훈련 과정에서 층에 있는 일부 뉴런을 랜덤하거 꺼서(뉴런의 출력을 0으로 만들어)
# 과대적합을 막는다

# 뉴런은 랜덤하게 드롭아웃되고 얼마나 많은 뉴런을 드롭할지는 하이퍼파라미터로 정한다

# 이전 층의 일부 뉴런이 랜덤하게 꺼지면 특정 뉴런에 과대하게 의존하는 것을 줄일 수 있고
# 모든 입력에 대해 주의를 기울여야 한다
# 일부 뉴런의 출력이 없을 수 있다는 것을 감안하면 이 신경망은 더 안정적인 예측을 만들 수 있을 것이다

# 또 다른 해석은 드롭아웃을 적용해 훈련하는 것이 마치 신경망을 앙상블 하는 것처럼 
# 상상할 수 있다 앙상블은 과대적합을 막아 주는 아주 좋은 기법이다

# 케라느에서는 드롭아웃을 keras.layers 패키지 아래 Dropout 클래스로 제공
# 어떤 층의 뒤에 드롭아웃을 두어 이 층의 출력을 랜덤하게 0으로 만든다
# 드롭아웃이 층처럼 사용되지만 훈련되는 모델 파라미터는 없다

# 30%정도를 드롭아웃하는 모델 생성
model = model_fn(keras.layers.Dropout(0.3))
model.summary()
# Model: "sequential_13"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten_13 (Flatten)        (None, 784)               0         
                                                                 
#  dense_26 (Dense)            (None, 100)               78500     
                                                                 
#  dropout (Dropout)           (None, 100)               0         
                                                                 
#  dense_27 (Dense)            (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 79510 (310.59 KB)
# Trainable params: 79510 (310.59 KB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# 은닉층 뒤에 추가된 드롭아웃 층은 훈련되는 모델 파리미터가 없다
# 또한 입력과 출력의 크기가 같다 
# 일부 뉴런의 출력을 0으로 만들지만 전체 출력 배열의 크기를 바꾸지는 않는다

# 훈련이 끝난 뒤에 평가나 예측을 수행할 때는 드롭아웃을 적용하지 말아야 한다
# 훈련된 모든 뉴런을 사용해야 올바른 예측을 수행할 수 있다
# 텐서플로와 케라스는 모델을 평가와 예측에 사용할 때는 자동으로 드롭아웃을 적용하지 않는다

# 훈련 손실과 검증 손실의 그래프를 출력
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 과대적합이 확실히 줄은 결과가 출력
# 열 번째 에포크 정도에서 검증 손실의 감소가 멈추지만 크게 상승하지 않고 어느 정도 유지
# 이 모델은 20번의 에포크 동안 훈련을 했기 때문에 결국 다시 과대적합이 되었다

# 에포크 횟수 10으로 지정하고 모델 훈련
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=10, verbose=0, validation_data=(val_scaled, val_target))

# 케라스 모델은 훈련된 모델의 파라미터를 저장하는 save_weights() 메서드를 제공
# 기본적으로 이 메서드는 텐서플로의 체크포인트 포맷으로 저장하지만 파일의 확장자가 .h5일 경우
# HDF5 포맷으로 저장
model.save_weights('model-weights.h5')

# 모델 구조와 모델 파라미터를 함께 저장하는 save() 메서드도 제공
# 기본적으로 이 메서드는 텐서플로의 SavedModel 포맷으로 저장하지만 파일의 확장자가 .h5일 경우
# HDF5 포맷으로 저장
model.save('model-whole.h5')

# 저장된 모델 확인
!ls -al *.h5

# 훈련을 하지 않은 새로운 모델을 만들고 model-weights.h5 파일에서 훈련된 모델 파라미터를 읽어서 사용
# load_weights() 메서드를 사용하려면 save_weights() 메서드로 저장했던 모델과 정확히 같은 구조를 가져야 한다
# 그렇지 않으면 에러 발생
model = model_fn(keras.layers.Dropout(0.3))
mode.load_weights('model-weights.h5')

# 검증 정확도 확인
# 케라스에서 예측을 수행하는 predict() 메서드는 사이킷런과 달리 샘플마다 10개의 클래스에 대한 확률을 반환
# 패현 MNIST 데이터셋이 다중 분류 문제이기 때문 (이진 분류 문제라면 양성 클래스에 대한 확률 하나만 반환)
# 케라스 모델에는 predict_classes() 메서드로 그냥 클래스를 예측해 주기도 하지만 사라질 예정이다
# 패션 MNIST 데이터셋에서 덜어낸 검증 세트의 샘플 개수는 12000개이기 떄뮨에 predict() 메서드는
# (12000, 10) 크기의 배열을 반환

# 모델 파라미터를 읽은 후 evaluate() 메서드를 사용하여 정확도를 출력할 수도 있다
# 하지만 evaluate() 메서드는 손실을 계산하기 위해 반드시 먼저 compile() 메서드를 실행해야 한다
# 여기에서는 새로운 데이터에 대해 정확도만 계산하면 되는 상황이라고 가정

# 10개 확률 중에 가장 큰 값의 인덱스를 골라 타깃 레이블과 비교하여 정확도를 계산
import numpy as np
val_labels = np.argmax(model.predict(val_scaled), axis=-1)
print(np.mean(val_labels == val_target))
# 0.88225

# argmax()
# 배열에서 가장 큰 값의 인덱스를 반환
# axis=-1은 배열의 마지막 차원을 따라 최댓값을 고른다
# 검증 세트는 2차원 배열이기 때문에 마지막 차원은 1이 된다

# armmax()로 고른 인덱스와 타깃을 비교한다
# 두 배열에서 각 위치의 값이 같으면 1 아니면 0이 된다
# 이를 평균하면 정확도가 된다

# 모델 전체를 파일에서 읽은 다음 검증 세트의 정확도 출력
model = keras.models.load_model('model-whole.h5')
model.evaluate(val_scaled, val_target)

# load_model() 함수는 모델 파라미터뿐만 아니라 모델 구조와 옵티마이저 상태까지 복원
# evaluate() 메서드를 사용 가능
# 텐서플로 2.3에서는 load_model() 함수의 버그 때문에 compile() 메서드를 호출해야 한다

# 콜백(callback)
# 훈련 과정 중간에 어떤 작업을 수행할 수 있게 하는 객체로 keras.callbacks 패키지 아래에 있는 클래스들
# fit() 메서드의 callbacks 매개변수에 리스트로 전달하여 사용
# ModelCheckpoint 콜백은 기본적으로 에포크마다 모델을 저장
# save_best_only=True 매개변수를 지정하여 가장 낮은 검증 점수를 만드는 모델을 저장할 수 있다

# 저장 파일 이름을 best-model.h5로 지정한 콜백 적용
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)

model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target),
    callbacks=[checkpoint_cb]
)

# 모델이 훈련한 후 best-model.h5에 최상의 검증 점수를 낸 모델이 저장
# 이 모델을 다시 읽어서 예측 수행
model = keras.models.load_model('best-model.h5')
model.evaluate(val_scaled, val_target)
# 375/375 [==============================] - 1s 2ms/step - loss: 0.3187 - accuracy: 0.8921
# [0.3186511695384979, 0.8920833468437195]

# 조기종료(early stopping)
# 과대적합이 커지기 시작하는 에포크 횟수부터는 훈련을 계속할 필요가 없다
# 이때 훈련을 중지하면 컴퓨터 자원과 시간을 아낄 수 있다
# 이렇게 과대적합이 시작되기 전에 훈련을 미리 중지하는 것을 조기 종료라고 부르며
# 딥러닝 분야에서 널리 사용한다

# 조기 종료는 훈련 에포크 횟수를 제한하는 역할이지만 모델이 과대적합되는 것을 막아 주기
# 때문에 규제 방법 중 하나로 생각할 수도 있다

# 케라스는 조기 종료를 위한 EarlyStopping 콜백을 제공한다
# 이 콜백의 patience 매개변수는 검증 점수가 향상되지 않더라도 참을 에포크 횟수로 지정한다
# 예를 들어 patience=2로 지정하면 2번 연속 검증 점수가 향상되지 않으면 훈련을 중지한다
# restore_best_weights 매개변수를 True로 지정하면 가장 낮은 검증 손실을 낸 모델 파라미터로 되돌린다

# EarlyStopping 콜백을 ModelCheckpoint 콜백과 함께 사용하면 가장 낮은 검증 손실의 모델을
# 파일에 저장하고 검증 손실이 다시 상승할 때 훈련을 중지할 수 있다
# 또한 훈련을 중지한 다음 현재 모델의 파라미터를 최상의 파라미터로 되돌린다

# 두 콜백을 사용한 코드
model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# 훈련을 마치고 나면 몇 번째 에포크에서 훈련이 중지되었는지 EarlyStopping 객체의 stopped_epoch 속성에서 확인 가능
print(early_stopping_cb.stopped_epoch)
# 8

# 에포크 횟수가 0부터 시작하기 때문에 8은 아홉 번째 에포크에서 훈련이 중지되었다는 것을 의미
# patience를 2로 지정했으므로 최상의 모델은 일곱 번째 에포크일 것이다

# 훈련 손실과 검증 손실 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 조기 종료 기법을 사용하면 에포크 횟수를 크게 지정해도 괜찮다


