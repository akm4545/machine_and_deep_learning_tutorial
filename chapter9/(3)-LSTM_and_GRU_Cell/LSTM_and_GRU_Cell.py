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

# 훈련 손실과 검증 손실 그래프 출력
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 그래프를 보면 기본 순환층보다 LSTM이 과대적합을 억제하면서 훈련을 잘 수행한 것으로 보인다
# 경우에 따라서는 과대적합을 더 강하게 제어할 필요가 있다

# 완전 연결 신경망과 합성곱 신경망에서는 Dropout 클래스를 사용해 드롭아웃을 적용했다
# 이를 통해 모델이 훈련 세트에 너무 과대적합되는 것을 막았다
# 순환층은 자체적으로 드롭아웃 기능을 제공한다
# SimpleRNN과 LSTM 클래스 모두 dropout 매개변수와 recurrent_dropout 매개변수를 가지고 있다

# dropout 매개변수는 셀의 입력에 드롭아웃을 적용하고 recurrent_dropout은 순환되는 은닉 상태에
# 드롭아웃을 적용한다

# 하지만 기술적인 문제로 인해 recurrent_dropout을 사용하면 GPU를 사용하여 모델을 훈련하지 못한다
# 이 때문에 모델의 훈련 속도가 크게 느려진다

# dropout만을 사용한 LSTM 클래스
# dropout 매개변수를 0.3으로 지정하여 30%의 입력을 드롭아웃
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

# 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-dropout-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model2.fit(train_seq, train_target, 
    epochs=100, batch_size=64, 
    validation_data=(val_seq, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb])
# Epoch 1/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 15s 41ms/step - accuracy: 0.5000 - loss: 0.6931 - val_accuracy: 0.5284 - val_loss: 0.6926
# Epoch 2/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 40ms/step - accuracy: 0.5322 - loss: 0.6922 - val_accuracy: 0.6206 - val_loss: 0.6910
# Epoch 3/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 38ms/step - accuracy: 0.5998 - loss: 0.6900 - val_accuracy: 0.6654 - val_loss: 0.6860
# Epoch 4/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.6404 - loss: 0.6800 - val_accuracy: 0.6962 - val_loss: 0.6397
# Epoch 5/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 40ms/step - accuracy: 0.6822 - loss: 0.6322 - val_accuracy: 0.7122 - val_loss: 0.6103
# Epoch 6/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.7074 - loss: 0.6040 - val_accuracy: 0.7332 - val_loss: 0.5894
# Epoch 7/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 39ms/step - accuracy: 0.7301 - loss: 0.5844 - val_accuracy: 0.7394 - val_loss: 0.5700
# Epoch 8/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 37ms/step - accuracy: 0.7450 - loss: 0.5624 - val_accuracy: 0.7594 - val_loss: 0.5514
# Epoch 9/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.7585 - loss: 0.5464 - val_accuracy: 0.7732 - val_loss: 0.5337
# Epoch 10/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 40ms/step - accuracy: 0.7619 - loss: 0.5314 - val_accuracy: 0.7688 - val_loss: 0.5175
# Epoch 11/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 41ms/step - accuracy: 0.7705 - loss: 0.5150 - val_accuracy: 0.7814 - val_loss: 0.5021
# Epoch 12/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 40ms/step - accuracy: 0.7830 - loss: 0.4993 - val_accuracy: 0.7888 - val_loss: 0.4883
# Epoch 13/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.7885 - loss: 0.4828 - val_accuracy: 0.7868 - val_loss: 0.4810
# Epoch 14/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 40ms/step - accuracy: 0.7906 - loss: 0.4755 - val_accuracy: 0.7890 - val_loss: 0.4713
# Epoch 15/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 40ms/step - accuracy: 0.7876 - loss: 0.4741 - val_accuracy: 0.7910 - val_loss: 0.4676
# Epoch 16/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 40ms/step - accuracy: 0.7961 - loss: 0.4621 - val_accuracy: 0.7922 - val_loss: 0.4595
# Epoch 17/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 38ms/step - accuracy: 0.7976 - loss: 0.4602 - val_accuracy: 0.7938 - val_loss: 0.4594
# Epoch 18/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 37ms/step - accuracy: 0.8009 - loss: 0.4515 - val_accuracy: 0.7940 - val_loss: 0.4537
# Epoch 19/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 22s 41ms/step - accuracy: 0.8017 - loss: 0.4474 - val_accuracy: 0.7940 - val_loss: 0.4505
# Epoch 20/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 40ms/step - accuracy: 0.8009 - loss: 0.4469 - val_accuracy: 0.7940 - val_loss: 0.4481
# Epoch 21/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 40ms/step - accuracy: 0.8056 - loss: 0.4365 - val_accuracy: 0.8010 - val_loss: 0.4459
# Epoch 22/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 40ms/step - accuracy: 0.8021 - loss: 0.4398 - val_accuracy: 0.7960 - val_loss: 0.4446
# Epoch 23/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 37ms/step - accuracy: 0.8155 - loss: 0.4269 - val_accuracy: 0.7912 - val_loss: 0.4465
# Epoch 24/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 22s 41ms/step - accuracy: 0.8050 - loss: 0.4328 - val_accuracy: 0.7990 - val_loss: 0.4421
# Epoch 25/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 41ms/step - accuracy: 0.8007 - loss: 0.4365 - val_accuracy: 0.8014 - val_loss: 0.4400
# Epoch 26/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 41ms/step - accuracy: 0.8040 - loss: 0.4355 - val_accuracy: 0.8002 - val_loss: 0.4442
# Epoch 27/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 41ms/step - accuracy: 0.8055 - loss: 0.4307 - val_accuracy: 0.8006 - val_loss: 0.4381
# Epoch 28/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 40ms/step - accuracy: 0.8033 - loss: 0.4375 - val_accuracy: 0.7902 - val_loss: 0.4446
# Epoch 29/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 41ms/step - accuracy: 0.8115 - loss: 0.4256 - val_accuracy: 0.7992 - val_loss: 0.4380
# Epoch 30/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 41ms/step - accuracy: 0.8088 - loss: 0.4235 - val_accuracy: 0.8016 - val_loss: 0.4362
# Epoch 31/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 21s 41ms/step - accuracy: 0.8106 - loss: 0.4228 - val_accuracy: 0.7960 - val_loss: 0.4376
# Epoch 32/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 13s 41ms/step - accuracy: 0.8106 - loss: 0.4211 - val_accuracy: 0.7992 - val_loss: 0.4371
# Epoch 33/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 20s 41ms/step - accuracy: 0.8129 - loss: 0.4193 - val_accuracy: 0.8012 - val_loss: 0.4398

# 검증 손실이 약간 향상되었다

# 훈련 손실과 검증 손실 그래프 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 훈련 손실과 검증 손실 간의 차이가 좁혀졌다

# 밀집층이나 합성곱 층처럼 순환층도 여러 개를 쌓지 않을 이유가 없다
# 순환층을 연결할 때는 한 가지 주의점이 있는데 순환층의 은닉 상태는 샘플의 마지막 타임스텝에 대한 은닉
# 상태만 다음 층으로 전달한다
# 하지만 순환층을 쌓게 되면 모든 순환층에 순차 데이터가 필요하다
# 따라서 앞쪽의 순환층이 모든 타임스텝에 대한 은닉 상태를 출력해야 한다
# 오직 마지막 순환층만 마지막 타임스텝의 은닉 상태를 출력해야 한다

# 케라스의 순환층에서 모든 타임스텝의 은닉 상태를 출력하려면 마지막을 제외한
# 다른 모든 순환층에서 return_sequences 매개변수를 True로 지정하면 된다

# 2개의 LSTM 층을 쌓았고 모두 드롭아웃을 0.3으로 지정
model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid'))

# 모델 구조 출력
model3.summary()
# Model: "sequential_6"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ embedding_6 (Embedding)              │ ?                           │     0 (unbuilt) │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ lstm_6 (LSTM)                        │ ?                           │     0 (unbuilt) │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ lstm_7 (LSTM)                        │ ?                           │     0 (unbuilt) │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_6 (Dense)                      │ ?                           │     0 (unbuilt) │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 0 (0.00 B)
#  Trainable params: 0 (0.00 B)
#  Non-trainable params: 0 (0.00 B)

# 모델 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model3.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-2rnn-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model3.fit(train_seq, train_target, 
    epochs=100, batch_size=64,
    validation_data=(val_seq, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb])

# Epoch 1/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 30s 82ms/step - accuracy: 0.4983 - loss: 0.6931 - val_accuracy: 0.5536 - val_loss: 0.6928
# Epoch 2/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - accuracy: 0.5319 - loss: 0.6926 - val_accuracy: 0.5966 - val_loss: 0.6918
# Epoch 3/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 79ms/step - accuracy: 0.5722 - loss: 0.6909 - val_accuracy: 0.6436 - val_loss: 0.6861
# Epoch 4/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - accuracy: 0.6152 - loss: 0.6793 - val_accuracy: 0.6876 - val_loss: 0.6296
# Epoch 5/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 79ms/step - accuracy: 0.6751 - loss: 0.6216 - val_accuracy: 0.7144 - val_loss: 0.5763
# Epoch 6/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - accuracy: 0.7189 - loss: 0.5703 - val_accuracy: 0.7296 - val_loss: 0.5498
# Epoch 7/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - accuracy: 0.7318 - loss: 0.5478 - val_accuracy: 0.7482 - val_loss: 0.5245
# Epoch 8/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 24s 76ms/step - accuracy: 0.7507 - loss: 0.5261 - val_accuracy: 0.7650 - val_loss: 0.5089
# Epoch 9/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 42s 78ms/step - accuracy: 0.7647 - loss: 0.5094 - val_accuracy: 0.7702 - val_loss: 0.4989
# Epoch 10/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 79ms/step - accuracy: 0.7744 - loss: 0.4970 - val_accuracy: 0.7738 - val_loss: 0.4912
# Epoch 11/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 79ms/step - accuracy: 0.7799 - loss: 0.4846 - val_accuracy: 0.7744 - val_loss: 0.4846
# Epoch 12/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 80ms/step - accuracy: 0.7823 - loss: 0.4802 - val_accuracy: 0.7830 - val_loss: 0.4757
# Epoch 13/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 79ms/step - accuracy: 0.7853 - loss: 0.4777 - val_accuracy: 0.7846 - val_loss: 0.4731
# Epoch 14/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 79ms/step - accuracy: 0.7880 - loss: 0.4710 - val_accuracy: 0.7834 - val_loss: 0.4680
# Epoch 15/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 25s 78ms/step - accuracy: 0.7919 - loss: 0.4668 - val_accuracy: 0.7846 - val_loss: 0.4660
# Epoch 16/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 24s 78ms/step - accuracy: 0.7890 - loss: 0.4653 - val_accuracy: 0.7756 - val_loss: 0.4739
# Epoch 17/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 24s 78ms/step - accuracy: 0.8004 - loss: 0.4495 - val_accuracy: 0.7608 - val_loss: 0.4989
# Epoch 18/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 78ms/step - accuracy: 0.7910 - loss: 0.4608 - val_accuracy: 0.7876 - val_loss: 0.4609
# Epoch 19/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 79ms/step - accuracy: 0.7938 - loss: 0.4602 - val_accuracy: 0.7902 - val_loss: 0.4614
# Epoch 20/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 78ms/step - accuracy: 0.7984 - loss: 0.4499 - val_accuracy: 0.7838 - val_loss: 0.4561
# Epoch 21/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 77ms/step - accuracy: 0.7995 - loss: 0.4497 - val_accuracy: 0.7918 - val_loss: 0.4527
# Epoch 22/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 40s 74ms/step - accuracy: 0.7998 - loss: 0.4453 - val_accuracy: 0.7932 - val_loss: 0.4483
# Epoch 23/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 42s 79ms/step - accuracy: 0.7953 - loss: 0.4507 - val_accuracy: 0.7890 - val_loss: 0.4590
# Epoch 24/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 41s 78ms/step - accuracy: 0.8064 - loss: 0.4380 - val_accuracy: 0.7950 - val_loss: 0.4490
# Epoch 25/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 24s 77ms/step - accuracy: 0.8052 - loss: 0.4446 - val_accuracy: 0.7854 - val_loss: 0.4612

# 일반적으로 순환층을 쌓으면 성능이 높아진다
# 이 예에서는 그리 큰 효과를 내지 못했다

# 손실 그래프 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show(0)

# 그래프를 보면 과대적합을 제어하면서 손실을 최대한 낮췄다

# GRU 구조
# GRU (Gated Recurrent Unit)의 약자
# LSTM을 간소화한 버전으로 생각할 수 있다
# 이 셀은 LSTM처럼 셀 상태를 계산하지 않고 은닉 상태 하나만 포함하고 있다

# GRU 셀에는 은닉 상태와 입력에 가중치를 곱하고 절편을 더하는 작은 셀이 3개 들어있다
# 2개는 시그모이드 활성화 함수를 사용하고 하나는 tanh 활성화 함수를 사용

# GRU셀은 LSTM보다 가중치가 적기 때문에 계산량이 적지만 LSTM 못지않은 좋은 성능을 낸다

# GRU 신경망 생성
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))

# 모델 구조 출력
model4.summary()
# Model: "sequential_3"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ embedding_3 (Embedding)              │ ?                           │     0 (unbuilt) │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ gru (GRU)                            │ ?                           │     0 (unbuilt) │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ ?                           │     0 (unbuilt) │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 0 (0.00 B)
#  Trainable params: 0 (0.00 B)
#  Non-trainable params: 0 (0.00 B)