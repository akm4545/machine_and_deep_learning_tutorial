# LSTM과 GRU
# 고급 순환층으로 SimpleRNN보다 계산이 훨씬 복잡하다
# 하지만 성능이 뛰어나기 때문에 순환 신경망에 많이 채택되고 있다

# 일반적으로 기본 순환층은 긴 시퀀스를 학습하기 어렵다
# 시퀀스가 길수록 순환되는 은닉 상태에 담긴 정보가 점차 희석되기 떄문이다
# 따라서 멀리 떨어져 있는 단어 정보를 인식하는 데 어려울 수 있다
# 이를 위해 LSTM과 GRU 셀이 발명되었다

# LSTM 구조
# Long Short-Term Memory의 약자
# 단기 기억을 오래 기억하기 위해 고안되었다
# 구조가 복잡하지만 기본 개념은 동일하다
# LSTM에는 입력과 가중치를 곱하고 절편을 더해 활성화 함수를 통과시키는 구조를 여러 개 가지고 있다
# 이런 계산 결과는 다음 타임스텝에 재사용된다

# 은닉 상태는 입력과 이전 타임스텝의 은닉 상태를 가중치에 곱한 후 
# 활성화 함수를 통과시켜 다음 은닉 상태를 만든다
# 이떄 기본 순환층과는 달리 시그모이드 활성화 함수를 사용한다
# 또 tanh 활성화 함수를 통과한 어떤 값과 곱해져서 은닉 상태를 만든다

# LSTM에는 순환되는 상태가 2개이다
# 은닉 상태 말고 셀 상태(cell state)ㄹ고 부르는 값이 있다
# 은닉 상태와 달리 셀 상태는 다음 층으로 전달되지 않고 LSTM 셀에서 순환만 되는 값이다

# 셀 상태를 계산하는 과정
# 먼저 입력과 은닉 상태를 또 다른 가중치에 곱한 다음 시그모이드 함수를 통과시킨다
# 그다음 이전 타임스텝의 셀 상태와 곱하여 새로운 셀 상태를 만든다
# 이 셀 상태가 tanh 함수를 통과하여 새로운 은닉 상태를 만드는데 기여한다

# LSTM은 마치 작은 셀을 여러 개 포함하고 있는 큰 셀 같다
# 중요한 것은 은닉 상태에 곱해지는 가중치와 셀 상태를 만드는데 사용하는 가중치가 다르다는 점이다
# 이 두 작은 셀은 각기 다른 기능을 위해 훈련된다

# 추가로 여기에 2개의 작은 셀이 더 추가되어 셀 상태를 만드는 데 기여한다
# 마찬가지로 입력과 은닉 상태를 각기 다른 가중치에 곱한 다음 하나는 시그모이드 함수를 통과시키고
# 다른 하나는 tanh 함수를 통과시킨다
# 그다음 두 결과를 곱한 후 이전 셀 상태와 더한다
# 이 결과가 최종적인 다음 셀 상태가 된다

# LSTM에는 총 4개의 셀이 있다

# 이 셀들을 삭제 게이트, 입력 게이트, 출력 게이트라고 부른다
# 삭제 게이트는 셀 상태에 있는 정보를 제거하는 역할을 하고 
# 입력 게이트는 새로운 정보를 셀 상태에 추가한다
# 출력 게이트를 통해서 이 셀 상태가 다음 은닉 상태로 출력된다

# LSTM 신경망 훈련
# IMDB 리뷰 데이터를 로드하고 훈련 세트와 검증 세트로 나눈다 
# 500개의 단어를 사용
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

# 케라스의 pad_sequences() 함수로 각 샘플의 길이를 100에 맞추고 부족할 때는 패딩을 추가
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

# LSTM 셀을 사용한 순환층 생성
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 모델 구조 출력
model.summary()
# Model: "sequential_2"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ embedding_1 (Embedding)              │ ?                           │     0 (unbuilt) │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ lstm (LSTM)                          │ ?                           │     0 (unbuilt) │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_2 (Dense)                      │ ?                           │     0 (unbuilt) │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 0 (0.00 B)
#  Trainable params: 0 (0.00 B)
#  Non-trainable params: 0 (0.00 B)

# 모델을 컴파일 하고 훈련
# 배치 크기 64, 에포크 횟수 100
# 체크포인트와 조기 종료
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_seq, train_target, 
    epochs=100, batch_size=64, 
    validation_data=(val_seq, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb])
# Epoch 1/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 17s 46ms/step - accuracy: 0.5282 - loss: 0.6928 - val_accuracy: 0.5744 - val_loss: 0.6919
# Epoch 2/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 45ms/step - accuracy: 0.5803 - loss: 0.6916 - val_accuracy: 0.5958 - val_loss: 0.6902
# Epoch 3/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 47ms/step - accuracy: 0.6090 - loss: 0.6894 - val_accuracy: 0.6216 - val_loss: 0.6867
# Epoch 4/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 48ms/step - accuracy: 0.6301 - loss: 0.6850 - val_accuracy: 0.6498 - val_loss: 0.6784
# Epoch 5/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 19s 42ms/step - accuracy: 0.6536 - loss: 0.6734 - val_accuracy: 0.6890 - val_loss: 0.6498
# Epoch 6/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 41ms/step - accuracy: 0.7005 - loss: 0.6301 - val_accuracy: 0.7182 - val_loss: 0.5899
# Epoch 7/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 44ms/step - accuracy: 0.7264 - loss: 0.5805 - val_accuracy: 0.7310 - val_loss: 0.5679
# Epoch 8/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 42ms/step - accuracy: 0.7393 - loss: 0.5618 - val_accuracy: 0.7468 - val_loss: 0.5476
# Epoch 9/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 22s 47ms/step - accuracy: 0.7500 - loss: 0.5436 - val_accuracy: 0.7510 - val_loss: 0.5342
# Epoch 10/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 44ms/step - accuracy: 0.7713 - loss: 0.5166 - val_accuracy: 0.7698 - val_loss: 0.5158
# Epoch 11/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 44ms/step - accuracy: 0.7729 - loss: 0.5100 - val_accuracy: 0.7696 - val_loss: 0.5047
# Epoch 12/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 45ms/step - accuracy: 0.7768 - loss: 0.4957 - val_accuracy: 0.7710 - val_loss: 0.4952
# Epoch 13/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 46ms/step - accuracy: 0.7866 - loss: 0.4812 - val_accuracy: 0.7838 - val_loss: 0.4825
# Epoch 14/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 44ms/step - accuracy: 0.7900 - loss: 0.4721 - val_accuracy: 0.7824 - val_loss: 0.4788
# Epoch 15/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 45ms/step - accuracy: 0.7963 - loss: 0.4640 - val_accuracy: 0.7886 - val_loss: 0.4663
# Epoch 16/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 15s 47ms/step - accuracy: 0.8033 - loss: 0.4498 - val_accuracy: 0.7910 - val_loss: 0.4608
# Epoch 17/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 46ms/step - accuracy: 0.8041 - loss: 0.4439 - val_accuracy: 0.7920 - val_loss: 0.4547
# Epoch 18/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 45ms/step - accuracy: 0.8062 - loss: 0.4395 - val_accuracy: 0.7884 - val_loss: 0.4552
# Epoch 19/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 46ms/step - accuracy: 0.8055 - loss: 0.4362 - val_accuracy: 0.7888 - val_loss: 0.4510
# Epoch 20/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 44ms/step - accuracy: 0.8098 - loss: 0.4314 - val_accuracy: 0.7978 - val_loss: 0.4441
# Epoch 21/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 43ms/step - accuracy: 0.8132 - loss: 0.4253 - val_accuracy: 0.8000 - val_loss: 0.4441
# Epoch 22/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 43ms/step - accuracy: 0.8103 - loss: 0.4225 - val_accuracy: 0.7922 - val_loss: 0.4432
# Epoch 23/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 44ms/step - accuracy: 0.8110 - loss: 0.4243 - val_accuracy: 0.7996 - val_loss: 0.4384
# Epoch 24/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 44ms/step - accuracy: 0.8090 - loss: 0.4274 - val_accuracy: 0.7904 - val_loss: 0.4436
# Epoch 25/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 43ms/step - accuracy: 0.8185 - loss: 0.4121 - val_accuracy: 0.7986 - val_loss: 0.4350
# Epoch 26/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 43ms/step - accuracy: 0.8131 - loss: 0.4184 - val_accuracy: 0.7976 - val_loss: 0.4437
# Epoch 27/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 45ms/step - accuracy: 0.8138 - loss: 0.4166 - val_accuracy: 0.7982 - val_loss: 0.4333
# Epoch 28/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 44ms/step - accuracy: 0.8134 - loss: 0.4169 - val_accuracy: 0.7982 - val_loss: 0.4330
# Epoch 29/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 45ms/step - accuracy: 0.8150 - loss: 0.4137 - val_accuracy: 0.8004 - val_loss: 0.4325
# Epoch 30/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 44ms/step - accuracy: 0.8132 - loss: 0.4111 - val_accuracy: 0.7970 - val_loss: 0.4359
# Epoch 31/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 43ms/step - accuracy: 0.8137 - loss: 0.4125 - val_accuracy: 0.8018 - val_loss: 0.4313
# Epoch 32/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 43ms/step - accuracy: 0.8172 - loss: 0.4062 - val_accuracy: 0.7978 - val_loss: 0.4310
# Epoch 33/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 42ms/step - accuracy: 0.8169 - loss: 0.4061 - val_accuracy: 0.7930 - val_loss: 0.4342
# Epoch 34/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 42ms/step - accuracy: 0.8122 - loss: 0.4088 - val_accuracy: 0.7960 - val_loss: 0.4438
# Epoch 35/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 44ms/step - accuracy: 0.8172 - loss: 0.4045 - val_accuracy: 0.8032 - val_loss: 0.4293
# Epoch 36/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 43ms/step - accuracy: 0.8275 - loss: 0.3899 - val_accuracy: 0.8018 - val_loss: 0.4290
# Epoch 37/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 44ms/step - accuracy: 0.8200 - loss: 0.3995 - val_accuracy: 0.7990 - val_loss: 0.4303
# Epoch 38/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 44ms/step - accuracy: 0.8180 - loss: 0.4017 - val_accuracy: 0.7930 - val_loss: 0.4333
# Epoch 39/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 45ms/step - accuracy: 0.8185 - loss: 0.4003 - val_accuracy: 0.8032 - val_loss: 0.4283
# Epoch 40/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 43ms/step - accuracy: 0.8125 - loss: 0.4068 - val_accuracy: 0.8018 - val_loss: 0.4300
# Epoch 41/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 14s 43ms/step - accuracy: 0.8164 - loss: 0.4008 - val_accuracy: 0.8012 - val_loss: 0.4290
# Epoch 42/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 43ms/step - accuracy: 0.8195 - loss: 0.4040 - val_accuracy: 0.8030 - val_loss: 0.4319    