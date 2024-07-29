# 텐서플로를 사용하면 합성곱, 패딩, 풀링 크기를 직접 계산할 필요가 없다
# 복잡한 계산은 케라스 API에 모두 위임하고 사용자는 직관적으로 
# 신경망을 설계할 수 있다

# 패션 MNIST 데이터를 불러오고 전처리
# 데이터 스케일을 0~255 사이에서 0~1 사이로 바꾸고 훈랸 세트와 검증 세트로 나눔
# 완전 연결 신경망에서는 입력 이미지를 밀집층에 연결하기 위해 일렬로 펼쳐야 했다
# 합성곱 신경망은 2차원 이미지를 그대로 사용하기 때문에 일렬로 펼치지 않는다

# 입력 이미지는 항상 깊이(채널) 차원이 있어야 한다
# 흑백 이미지의 경우 채널 차원이 없는 2차원 배열이라 Conv2D 층을 사용하기 위해
# 마지막에 이 채널 차원을 추가해야 한다

# 넘파이 reshape() 메서드를 사용해 전체 배열 차원을 그대로 유지하면서 마지막에 차원 추가
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 케라스의 Sequential 클래스를 사용해 구조 정의
# 합성곱 층인 Conv2D 추가
# 이 클래스는 다른 층 클래스와 마찬가지로 keras.layers 패키지 아래에 있다
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))

# 해당 합성곱은 32개의 필터를 사용한다
# 커널의 크기는 (3, 3)이고 렐루 활성화 함수와 세임 패딩을 사용
# 신경망 모델의 첫 번째 층에서 입력의 차원을 지정해 주어야 한다
# 패션 MNIST 이미지를 (28, 28)에서 (28, 28, 1)로 변경했으므로 input_shape 매개변수를 이 값으로 지정

# 풀링 층은 keras.layers 패키지 아래에 MaxPooling2D(최대 풀링), AveragePooling2D(평균 풀링)
# 클래스로 제공
# Conv2D 클래스의 kernel_size 처럼 가로세로 크기가 같으면 정수 하나로 지정할 수 있다
model.add(keras.layers.MaxPooling2D(2))

# 세임 패딩을 적용했기 떄문에 합성곱 층에서 출력된 특성 맵의 가로세로 크기는 입력과 동일하다
# 그 다음 (2, 2) 풀링을 적용했으므로 특성 맵의 크기는 절반으로 줄어든다
# 합성곱 층에서 32개의 필터를 사용했기 때뭉네 이 특성 맵의 깊이는 32가 된다
# 따라서 최대 풀링을 통과한 특성 맵의 크기는 (14, 14, 32)가 될 것이다

# 두 번째 합성곱-풀링 층 추가
# 필터 개수 64개
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

# (밀집) 출력층에서 확률 계산을 위해 특성 맵을 일렬로 펼쳐야 한다
# Flatten클래스 -> Dense은닉층 -> Dropout 층 (과대적합 방지) -> Dense 출력층의 순서대로 구상
model.add(keras.layers.Flatten())
# 100개의 뉴런을 사용하고 활성화 함수는 렐루 함수 사용
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
# 10개의 클래스를 분류하는 다중 분류 문제이므로 마지막 층의 활성화 함수는 소프트맥스 사용
model.add(keras.layers.Dense(10, activation='softmax'))

# 모델 구조 출력
model.summary()
# Model: "sequential_3"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d_4 (Conv2D)           (None, 28, 28, 32)        320       
                                                                 
#  max_pooling2d_3 (MaxPoolin  (None, 14, 14, 32)        0         
#  g2D)                                                            
                                                                 
#  conv2d_5 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                 
#  max_pooling2d_4 (MaxPoolin  (None, 7, 7, 64)          0         
#  g2D)                                                            
                                                                 
#  flatten_1 (Flatten)         (None, 3136)              0         
                                                                 
#  dense_2 (Dense)             (None, 100)               313700    
                                                                 
#  dropout_1 (Dropout)         (None, 100)               0         
                                                                 
#  dense_3 (Dense)             (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 333526 (1.27 MB)
# Trainable params: 333526 (1.27 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# 층의 구성을 그림으로 표현해 주는 plot_model() 함수도 있다
# 이 함수는 keras.utils 패키지에 있다
keras.utils.plot_model(model)

# plot_model() 함수의 show_shapes 매개변수를 True로 설정하면 이 그림에 입력과 출력의
# 크기를 표시해 준다
# to_file 매개변수에 파일 이름을 지정하면 출력한 이미지를 파일로 저장한다
# dpi 매개변수로 해상도를 지정할 수도 있다
keras.utils.plot_model(model, show_shapes=True)

# 케라스 API의 장점은 딥러닝 모델의 종류나 구성 방식에 상관없이 컴파일 훈련 과정이
# 같다는 점이다

# Adam 옵티마이저 사용, ModelCheckpoint 콜백, EarlyStopping 콜백을 사용하여 조기 종료 기법 구현
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])
# Epoch 1/20
# 1500/1500 [==============================] - 71s 47ms/step - loss: 0.5255 - accuracy: 0.8123 - val_loss: 0.3257 - val_accuracy: 0.8801
# Epoch 2/20
# 1500/1500 [==============================] - 68s 45ms/step - loss: 0.3478 - accuracy: 0.8739 - val_loss: 0.2773 - val_accuracy: 0.8961
# Epoch 3/20
# 1500/1500 [==============================] - 73s 49ms/step - loss: 0.2994 - accuracy: 0.8921 - val_loss: 0.2548 - val_accuracy: 0.9067
# Epoch 4/20
# 1500/1500 [==============================] - 72s 48ms/step - loss: 0.2702 - accuracy: 0.9007 - val_loss: 0.2412 - val_accuracy: 0.9093
# Epoch 5/20
# 1500/1500 [==============================] - 69s 46ms/step - loss: 0.2439 - accuracy: 0.9103 - val_loss: 0.2410 - val_accuracy: 0.9092
# Epoch 6/20
# 1500/1500 [==============================] - 67s 45ms/step - loss: 0.2230 - accuracy: 0.9180 - val_loss: 0.2351 - val_accuracy: 0.9158
# Epoch 7/20
# 1500/1500 [==============================] - 71s 47ms/step - loss: 0.2069 - accuracy: 0.9236 - val_loss: 0.2218 - val_accuracy: 0.9175
# Epoch 8/20
# 1500/1500 [==============================] - 71s 47ms/step - loss: 0.1919 - accuracy: 0.9285 - val_loss: 0.2252 - val_accuracy: 0.9196
# Epoch 9/20
# 1500/1500 [==============================] - 76s 51ms/step - loss: 0.1777 - accuracy: 0.9323 - val_loss: 0.2291 - val_accuracy: 0.9175

# 훈련 세트의 정확도가 이전보다 훨씬 좋아졌다

# 손실 그래프를 그려서 조기 종료가 잘 이루어졌는지 확인
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# EarlyStopping 클래스의 restore_best_weights 매개변수를 True로 지정했으므로
# 현재 model 객체가 최적의 모델 파라미터로 복원되어 있다

# 검증 세트 성능 평가
model.evaluate(val_scaled, val_target)
# 375/375 [==============================] - 5s 12ms/step - loss: 0.2218 - accuracy: 0.9175
# [0.22181083261966705, 0.9175000190734863]

# predict() 메서드를 사용해 새로운 데이터에 대한 예측을 만들기
# 검증 세트의 첫 번째 샘플을 처음 본 이미지라고 가정

# 첫 번째 샘플 이미지 출력
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()
# 핸드백 이미지

# 예측 확률 출력
preds = model.predict(val_scaled[0:1])
print(preds)
# [[2.2905691e-11 6.5928788e-15 5.3693714e-13 1.6925417e-12 2.0883654e-11
#   1.5328136e-11 7.3311496e-11 6.5840389e-13 1.0000000e+00 2.7428997e-12]]

# 케라스의 fit(), predict(), evaluate() 메서드 모두 입력의 첫 번째 차원이 배치 차원일 것으로 기대한다
# 따라서 샘플 하나를 전달할 때 (28, 28, 1)이 아니라 (1, 28, 28, 1)크기를 전달해야 한다
# 배열 슬라이싱은 인덱싱과 다르게 선택된 요소가 하나이더라도 전체 차원이 유지되어 (1, 28, 28, 1) 크기를 만든다

# 출력 결과를 보면 아홉 번째 값이 1이고 다른 값은 거의 0에 가깝다

# 막대 그래프로 출력
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

# 파이썬에서 레이블을 다루기 위해 리스트로 저장
classes = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

# 클래스 리스트가 있으면 레이블을 출력하기 쉽다
# preds 배열에서 가장 큰 인덱스를 찾아 classes 리스트의 인덱스로 사용하면 된다
import numpy as np
print(classes[np.argmax(preds)])
# 가방

# 테스트 세트로 합성곱 신경망의 일반화 성능 측정
# 즉 이 모델을 실전에 투입했을 때 얻을 수 잇는 예상 성능 측정

# 픽셀값 전처리
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0
model.evaluate(test_scaled, test_target)
# 313/313 [==============================] - 5s 14ms/step - loss: 0.2381 - accuracy: 0.9148
# [0.23806720972061157, 0.9147999882698059]

# 실전 투입 시 91%의 성능을 기대할 수 있다
