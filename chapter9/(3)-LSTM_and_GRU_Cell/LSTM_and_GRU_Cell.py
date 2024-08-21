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