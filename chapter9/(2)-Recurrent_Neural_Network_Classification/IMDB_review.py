# 자연어 처리(natural language processing, NLP)
# 컴퓨터를 사용해 인간의 언어를 처리하는 분야
# 대표적엔 세부 분야로는 음성 인식, 기계 번역, 감성 분석 들이 있다
# IMDB 리뷰를 감상평에 따라 분류하는 작업은 감성 분석에 해당한다

# 자연어 처리 분야에서는 훈련 데이터를 종종 말뭉치(corpus)라고 부른다
# 예를 들어 IMDB 리뷰 데이터셋이 하나의 말뭉치이다

# 텍스트 자체를 신경망에 전달하지는 않는다
# 컴퓨터에서는 처리하는 모든 것은 어떤 숫자 데이터이다
# 이미지는 정수 픽셀값으로 이루어져 있어 특별한 변환을 하지 않는다

# 텍스트 데이터의 경우 단어를 숫자 데이터로 바꾸는 일반적인 방법은 데이터에 등장하는
# 단어마다 고유한 정수를 부여하는 것이다
# 각 단어를 하나의 정수에 매핑하고 동일한 단어는 동일한 정수에 매핑한다
# 단어에 매핑되는 정수는 단어의 의미나 크기와 관련이 없다
# 정수값 사이에는 어떤 관계도 없다

# 일반적으로 영어 문장은 모두 소문자로 바꾸고 구둣점을 삭제한 다음 공백을 기분으로 분리한다
# 이렇게 분리된 단어를 토큰(token)이라고 부른다
# 하나의 샘플은 여러 개의 토큰으로 이루어져 있고 1개의 토큰이 하나의 타임스텝에 해당한다
# 간단한 문제라면 영어 말뭉치에서 토큰을 단어와 같게 봐도 좋다

# 한글 문장은 다른데 한글은 조사가 발달되어 있기 때문에 공백으로 나누는 것만으로는 부족하다
# 일반적으로 한글은 형태소 분석을 통해 토큰을 만든다

# 토큰에 할당하는 정수 중에 몇 개는 특정한 용도로 예약되어 있는 경우가 많다
# 예를 들어 0은 패딩, 1은 문장의 시작, 2는 어휘 사전에 없는 토큰을 나타낸다

# 어휘 사전은 훈련 세트에서 고유한 단어를 뽑아 만든 목록이다
# 예를 들어 테스트 세트 안에 어휘 사전에 없는 단어가 있다면 2로 변환하여 신경망 모델에 주입한다

# 실제 IMDB 리뷰 데이터셋은 영어로 된 문장이지만 텐서플로에는 이미 정수로 바꾼 데이터가 포함되어 있다

# tensorflow.keras.datasets 패키지 아래 imdb 모듈 임포트
# 전체 데이터셋에서 가장 자주 등장하는 단어 300개만 사용
# load_data 함수의 num_words 매개변수를 300으로 지정
from tensorflow.keras.datasets import imdb

(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=300)

# 훈련 세트와 테스트 세트의 크기 확인
print(train_input.shape, test_input.shape)
# (25000,) (25000,)

# IMDB 리뷰 텍스트는 길이가 제각각이다 
# 따라서 고정 크기의 2차원 배열에 담기 보다는 리뷰마다 별도의 파이썬 리스트로 담아야 메모리를
# 효율적으로 사용할 수 있다

# train_inpit: [리뷰1,       리뷰2, 리뷰3, ...] 넘파이 배열
#               파이썬 리스트

# 넘파이 배열은 정수나 실수 외에도 파이썬 객체를 담을 수 있다

# 첫 번쨰 리뷰의 길이 출력
print(len(train_input[0]))
# 218
# 첫 번째 리뷰의 길이는 218개의 토큰

# 두 번째 리뷰의 길이 출력
print(len(train_input[1]))
# 189

# 하나의 리뷰가 하나의 샘플이 된다

# 첫 번째 리뷰의 내용 출력
print(train_input[0])
# [1, 14, 22, 16, 43, 2, 2, 2, 2, 65, 2, 2, 66, 2, 4, 173, 36, 256, 5, 25, 100, 43, 2, 112, 50, 2, 2, 9, 35, 2, 284, 5, 150, 4, 172, 112, 167, 2, 2, 2, 39, 4, 172, 2, 2, 17, 2, 38, 13, 2, 4, 192, 50, 16, 6, 147, 2, 19, 14, 22, 4, 2, 2, 2, 4, 22, 71, 87, 12, 16, 43, 2, 38, 76, 15, 13, 2, 4, 22, 17, 2, 17, 12, 16, 2, 18, 2, 5, 62, 2, 12, 8, 2, 8, 106, 5, 4, 2, 2, 16, 2, 66, 2, 33, 4, 130, 12, 16, 38, 2, 5, 25, 124, 51, 36, 135, 48, 25, 2, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 2, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 2, 5, 2, 36, 71, 43, 2, 2, 26, 2, 2, 46, 7, 4, 2, 2, 13, 104, 88, 4, 2, 15, 297, 98, 32, 2, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 2, 26, 2, 5, 144, 30, 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 2, 88, 12, 16, 283, 5, 16, 2, 113, 103, 32, 15, 16, 2, 19, 178, 32]

# 텐서플로에 있는 IMDB 리뷰 데이터는 이미 정수로 변환되어 있다
# 앞서 num_words=300으로 지정했기 떄문에 어휘 사전에는 300개의 단어만 들어가 있다
# 따라서 어휘 사전에 없는 단어는 모두 2로 표시되어 나타난다
# imdb.load_data 함수는 전체 어휘 사전에 있는 단어를 등장 횟수 순서대로 나열한 다음 가장 많이
# 등장한 300개의 단어를 선택한다

# 타깃 데이터 출력
print(train_target[:20])
# [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]

# 해결할 문제는 리뷰가 긍정인지 부정인지를 판단하는 이진 분류 문제이다
# 타깃값은 0(부정)과 1(긍정)fh sksndjwlsek

# 훈련 세트에서 검증 세트 추출 검증 세트의 크기는 20%
from sklearn.model_selection import train_test_split

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)

# 평균적인 리뷰의 길이, 가장 짧은 리뷰의 길이, 가장 긴 리뷰의 길이를 확인하기 위해
# 각 리뷰의 길이를 계산해 넘파이 배열에 저장
# 넘파이 리스트 내포를 사용해 train_input의 원소를 순회하면서 길이 측정
import numpy as np

lengths = np.array([len(x) for x in train_input])

# 넘파이 mean 함수와 median 함수를 사용해 리뷰의 길이의 평균과 중간값 출력
print(np.mean(lengths), np.median(lengths)) 
# 239.00925 178.0
# 리뷰의 평균 단어 개수는 239개
# 중간값이 178개 

# 리뷰 테이터 단어 개수의 히스토그램
import matplotlib.pyplot as plt

plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()
# 데이터의 단어 개수가 한쪽으로 치우쳐져 있다
# 대부분의 리뷰 길이는 300 미만이다
# 평균이 중간값보다 높은 이유는 오른쪽 끝에 아주 큰 데이터가 있기 때문이다

# 리뷰는 대부분 짧아서 100개의 단어만 사용
# 100개의 단어보다 작은 리뷰는 길이를 100에 맞추기 위해 패딩이 필요하다
# 보통 패딩을 나타내는 토큰으로는 0을 사용한다

# 수동으로 훈련 세트에 있는 20000개의 리뷰를 순회하면서 길이가 100이 되도록
# 잘라내거나 0으로 패딩 할 수도 있지만 
# 케라스는 시퀀스 데이터의 길이를 맞추는 pad_sequences 함수를 제공한다

# 길이를 100으로 맞추기
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
# 100보다 길면 잘라내고 짧은 경우는 0으로 패딩

# train_seq의 크기 출력
print(train_seq.shape)
# (20000, 100)
# 20000 = 샘플 개수, 100 = 토큰(타임스텝) 개수

# train_seq에 있는 첫 번째 샘플을 출력
print(train_seq[0])
# [ 10   4  20   9   2   2   2   5  45   6   2   2  33 269   8   2 142   2
#    5   2  17  73  17 204   5   2  19  55   2   2  92  66 104  14  20  93
#   76   2 151  33   4  58  12 188   2 151  12 215  69 224 142  73 237   6
#    2   7   2   2 188   2 103  14  31  10  10   2   7   2   5   2  80  91
#    2  30   2  34  14  20 151  50  26 131  49   2  84  46  50  37  80  79
#    6   2  46   7  14  20  10  10   2 158]
# 이 샘플의 앞뒤에 패딩값 0이 없는것으로 보아 100 보다는 길었을 것이다

# 원본 샘플의 끝을 출력
print(train_input[0][-10:])
# [6, 2, 46, 7, 14, 20, 10, 10, 2, 158]
# 결과를 비교해 보면 샘플의 길이가 넘친 샘플의 앞부분이 잘렸다는것을 알 수 있다

# pad_sequences 함수는 기본적으로 maxlen보다 긴 시퀀스의 앞부분을 자른다
# 이렇게 하는 이뉴는 일반적으로 시퀀스의 뒷부분의 정보다 더 유용하리라 기대하기 때문이다
# 영화 리뷰 데이터를 생각해 보면 리뷰 끝에 뭔가 결정적인 소감을 말할 가능성이 높다고 볼 수 있다
# 만약 시퀀스의 뒷부분을 잘라내고 싶다면 pad_sequences 함수의 truncating 매개변수의 값을
# 기본값 pre가 아닌 post로 바꾸면 된다

# train_seq의 6번째 샘플 출력
print(train_seq[5])
# [  0   0   0   0   1   2 195  19  49   2   2 190   4   2   2   2 183  10
#   10  13  82  79   4   2  36  71 269   8   2  25  19  49   7   4   2   2
#    2   2   2  10  10  48  25  40   2  11   2   2  40   2   2   5   4   2
#    2  95  14 238  56 129   2  10  10  21   2  94   2   2   2   2  11 190
#   24   2   2   7  94 205   2  10  10  87   2  34  49   2   7   2   2   2
#    2   2 290   2  46  48  64  18   4   2]
# 이 샘플의 길이는 100이 안되므로 패딩이 추가되어 있다

# 같은 이유로 패딩 토큰은 시퀀스의 뒷부분이 아니라 앞부분에 추가된다
# 시퀀스의 마지막에 있는 단어가 셀의 은닉 상태에 가장 큰 영향을 미치게 되므로 마지막에
# 패딩을 추가하는 것은 일반적으로 선호하지 않는다
# 원한다면 pad_sequences 함수의 padding 매개변수의 기본값인 pre를 post로 바꾸면
# 샘플의 뒷부분에 패딩을 추가할 수 있다

# 검증 세트의 길이도 100으로 제한
val_seq = pad_sequences(val_input, maxlen=100)

# 케라스는 여러 종류의 순환층 클래스를 제공한다
# 구중에 가장 간단한 것은 SimpleRNN 클래스이다
# IMDB 리뷰 분류 문제는 이진 분류이므로 마지막 출력층은 1개의 뉴런을 가지고 시그모이드 활성화 함수를
# 사용해야 한다
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape=(100, 300)))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# SimpleRNN의 첫 번째 매개변수에는 사용할 뉴런의 개수를 지정
# input_shape에 입력 차원을 (100, 300)으로 지정
# 첫 번째 차원이 100인것은 샘플의 길이를 100으로 지정했기 때문이다

# 순환층도 활성화 함수를 사용해야 한다
# SimpleRNN 클래스의 activation 매개변수의 기본값은 tanh로 하이퍼볼릭 탄젠트 함수를 사용

# 전처리를 거친 데이터에는 한 가지 큰 문제가 있다
# 토큰을 정수로 변환한 데이터를 신경망에 주입하면 큰 정수가 큰 활성화 출력을 만든다
# 하지만 정수 사이에는 어떤 관련이 없다
# 20번 토큰을 10번 토큰보다 더 중요시해야 할 이유가 없다

# 정숫값에 있는 크기 속성을 없애고 각 정수를 고유하게 표현하는 방법은 원-핫 인코딩이다
# 예를 들어 train_seq[0]의 첫 번째 토큰인 10을 원-핫 인코딩으로 바꾸면 다음과 같다
# 0 0 0 0 0 0 0 0 0 0 1 0 ... 0
# 열한 번째 우너소만 1이고 나머지는 모두 0인 배열이다

# imdb.load_data 함수에서 300개의 단어만 사용하도록 지정했기 때문에 고유한 단어는 모두 300개이다
# 즉 훈련 데이터에 포함될 수 있는 정숫값의 범위는 0(패딩 토큰)에서 299까지이다
# 따라서 이 범위를 원-핫 인코딩으로 표현하려면 배열의 길이가 300이어야 한다

# 300개 중에 하나만 1이고 나머지는 모두 0으로 만들어 정수 사이에 있던 크기 속성을 없애는
# 원-핫 인코딩을 사용한다

# 케라스에는 원-핫 인코딩을 위한 유틸리티를 제공한다
# keras.utils 패키지 아래에 to_categorical 함수이다

# 원-핫 인코딩 배열 생성
train_oh = keras.utils.to_categorical(train_seq)

# 배열의 크기 출력
print(train_oh.shape)
# (20000, 100, 300)
# 정수 하나마다 모두 300차원의 배열로 변경됐기 때문에 (20000, 100) 크기의 train_seq가
# (20000,    100,            300) 크기의 train_oh로 바뀌었다
#  리뷰개수   리뷰 추출 단어   사용하는 단어 

# 이렇게 샘플 데이터의 크기가 1차원 정수 배열(100, )에서 2차원 배열(100, 300)로 바꿔야 하므로
# SimpleRNN 클래스의 input_shape 매개변수의 값을 (100, 300)으로 지정했다

# train_oh의 첫 번째 샘플의 첫 번째 토큰 10이 잘 인코딩되었는지 출력
print(train_oh[0][0][:12])
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]

# 나머지 원소가 모두 0인지 sum 함수로 모든 원소의 값을 더해서 1이 되는지 확인
print(np.sum(train_oh[0][0]))
# 1.0

# val_seq 원-핫 인고딩 전처리
val_oh = keras.utils.to_categorical(val_seq)

# 모델 구조 출력
model.summary()
# Model: "sequential_3"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ simple_rnn_3 (SimpleRNN)             │ (None, 8)                   │           2,472 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ (None, 1)                   │               9 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 2,481 (9.69 KB)
#  Trainable params: 2,481 (9.69 KB)
#  Non-trainable params: 0 (0.00 B)

# SimpleRNN에 전달할 샘플의 크기는 (100, 300)이지만 이 순환층은 마지막 타임스텝의 은닉 상태만
# 출력한다
# 이 때문에 출력 크기가 순환층의 뉴런 개수와 동일한 8이다

# 입력 토큰은 300차원의 원-핫 인코딩 배열이다
# 이 배열의 순환층의 뉴런 8개와 완전히 연결되기 떄문에 총 300 * 8 = 2400개의 가중치가 있다
# 순환층의 은닉 상태는 다시 다음 타임스텝에 사용되기 위해 또 다른 가중치와 곱해진다
# 이 은닉 상태도 순환층의 뉴런과 완전히 연결되기 떄문에 8(은닉 상태 크기) * 8(뉴런 개수) = 64개의 가중치가 필요하다
# 마지막으로 뉴런마다 하나의 절편이 있다 
# 따라서 모두 2400 + 64 + 8 = 2472개의 모델 파라미터가 필요하다

# 기본 RMSprop의 학습률 0.001을 사용하지 않기 위해 별도의 RMSprop 객체를 
# 만들어 학습률을 0.0001로 지정 
# 에포크 횟수를 100으로 늘리고 배치 크기는 64개로 설정
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', save_best_only=True)
# 케라스 버전 변경으로 인한 변경
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    train_oh, 
    train_target, 
    epochs=100, 
    # batch_size=64,
    # colab 램 리소스 이슈
    # 배치 사이즈 줄여도 해결되지 않음 
    # 학습 데이터 줄여야 할것 같음
    batch_size=32, 
    validation_data=(val_oh, val_target),
    callbacks=[checkpoint_cb, early_stopping_cb]
)

# 코랩 자원 이슈로 토큰 개수 70개로 줄임
# Epoch 1/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 12s 16ms/step - accuracy: 0.5101 - loss: 0.7092 - val_accuracy: 0.5324 - val_loss: 0.6940
# Epoch 2/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.5351 - loss: 0.6908 - val_accuracy: 0.5540 - val_loss: 0.6860
# Epoch 3/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 10s 17ms/step - accuracy: 0.5571 - loss: 0.6834 - val_accuracy: 0.6070 - val_loss: 0.6654
# Epoch 4/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 17ms/step - accuracy: 0.6093 - loss: 0.6588 - val_accuracy: 0.6374 - val_loss: 0.6395
# Epoch 5/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 20s 17ms/step - accuracy: 0.6504 - loss: 0.6282 - val_accuracy: 0.6614 - val_loss: 0.6213
# Epoch 6/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 23s 21ms/step - accuracy: 0.6743 - loss: 0.6111 - val_accuracy: 0.6800 - val_loss: 0.6053
# Epoch 7/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 19s 18ms/step - accuracy: 0.6893 - loss: 0.5976 - val_accuracy: 0.6888 - val_loss: 0.5929
# Epoch 8/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 19s 17ms/step - accuracy: 0.7025 - loss: 0.5809 - val_accuracy: 0.7022 - val_loss: 0.5819
# Epoch 9/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 19s 14ms/step - accuracy: 0.7083 - loss: 0.5761 - val_accuracy: 0.7072 - val_loss: 0.5751
# Epoch 10/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 17ms/step - accuracy: 0.7192 - loss: 0.5642 - val_accuracy: 0.7136 - val_loss: 0.5702
# Epoch 11/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 20s 16ms/step - accuracy: 0.7240 - loss: 0.5584 - val_accuracy: 0.7132 - val_loss: 0.5621
# Epoch 12/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 9s 14ms/step - accuracy: 0.7319 - loss: 0.5482 - val_accuracy: 0.7210 - val_loss: 0.5560
# Epoch 13/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 12s 17ms/step - accuracy: 0.7271 - loss: 0.5507 - val_accuracy: 0.7218 - val_loss: 0.5522
# Epoch 14/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 20s 17ms/step - accuracy: 0.7307 - loss: 0.5454 - val_accuracy: 0.7244 - val_loss: 0.5478
# Epoch 15/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 20s 17ms/step - accuracy: 0.7375 - loss: 0.5381 - val_accuracy: 0.7268 - val_loss: 0.5452
# Epoch 16/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 21s 17ms/step - accuracy: 0.7379 - loss: 0.5359 - val_accuracy: 0.7218 - val_loss: 0.5440
# Epoch 17/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 9s 14ms/step - accuracy: 0.7403 - loss: 0.5320 - val_accuracy: 0.7316 - val_loss: 0.5421
# Epoch 18/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 15ms/step - accuracy: 0.7451 - loss: 0.5311 - val_accuracy: 0.7242 - val_loss: 0.5408
# Epoch 19/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 17ms/step - accuracy: 0.7437 - loss: 0.5260 - val_accuracy: 0.7288 - val_loss: 0.5372
# Epoch 20/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 20s 16ms/step - accuracy: 0.7506 - loss: 0.5181 - val_accuracy: 0.7298 - val_loss: 0.5363
# Epoch 21/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 9s 15ms/step - accuracy: 0.7460 - loss: 0.5219 - val_accuracy: 0.7296 - val_loss: 0.5357
# Epoch 22/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 17ms/step - accuracy: 0.7482 - loss: 0.5203 - val_accuracy: 0.7360 - val_loss: 0.5355
# Epoch 23/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 12s 20ms/step - accuracy: 0.7451 - loss: 0.5264 - val_accuracy: 0.7364 - val_loss: 0.5324
# Epoch 24/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 19s 17ms/step - accuracy: 0.7508 - loss: 0.5201 - val_accuracy: 0.7364 - val_loss: 0.5329
# Epoch 25/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 17ms/step - accuracy: 0.7536 - loss: 0.5163 - val_accuracy: 0.7352 - val_loss: 0.5320
# Epoch 26/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 19s 14ms/step - accuracy: 0.7463 - loss: 0.5197 - val_accuracy: 0.7350 - val_loss: 0.5318
# Epoch 27/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 17ms/step - accuracy: 0.7472 - loss: 0.5194 - val_accuracy: 0.7370 - val_loss: 0.5296
# Epoch 28/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 17ms/step - accuracy: 0.7480 - loss: 0.5176 - val_accuracy: 0.7370 - val_loss: 0.5295
# Epoch 29/100
# 625/625 ━━━━━━━━━━━━━━━━━━━━ 11s 17ms/step - accuracy: 0.7519 - loss: 0.5129 - val_accuracy: 0.7370 - val_loss: 0.5282
# Epoch 30/100
# 책 예제에서는 80%의 정확도

# 매우 뛰어난 성능은 아니지만 감상평을 분류하는 데 어느 정도 성과를 내고 있다고 판단할 수 있다

# 훈련 손실과 검증 손실을 그래프로 출력
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 훈련 손실은 꾸준히 감소하고 있지만 검증 손실은 대략 스무 번째 에포크에서 감소가 둔해지고 있다

# 순환 신경망을 훈련시켜서 IMDB 리뷰 데이터를 긍정과 부정으로 분류하는 작업을 위해
# 입력 데이터를 원-핫 인코딩으로 변환했다
# 원-핫 인코딩의 단점은 입력 데이터가 엄청 커진다는 것이다

# train_seq 배열과 train_oh 배열의 nbytes 속성 출력 
print(train_seq.nbytes, train_oh.nbytes)

# 토큰 1개를 300차원으로 늘렸기 때문에 대략 300배가 커진다
# 이는 훈련 데이터가 커질수록 더 문제가 된다

# 단어 임베딩(word embedding)
# 순환 신경망에서 텍스트를 처리할 때 즐겨 사용하는 방법
# 단어 임베딩은 각 단어를 고정된 크기의 실수 벡터로 바꾸어 준다

# 예시 
# Cat의 단어 임베딩 벡터
# [0.2, 0.1, 1.3, 0.8, 0.2, 0.4, 1.1, 0.9, 0.2, 0.1]

# 단어 임베딩으로 만들어진 벡터는 원-핫 인코딩된 벡터보다 훨씬 의미 있는 값으로
# 채워져 있기 떄문에 자연어 처리에서 더 좋은 성능을 내는 경우가 많다
# 단어 임베딩 벡터를 만드는 층은 케라스의 kerasa.layers 패키지 아래 Embedding 클래스로 임베딩 기능 제공
# 이 클래스를 다른 층처럼 모델에 추가하면 처음에는 모든 벡터가 랜덤하게 초기화되지만 훈련을
# 통해 데이터에서 좋은 단어 임베딩을 학습한다

# 단어 임베딩의 장점은 입력으로 정수 데이터를 받는다는 것이다
# 즉 원-핫 인코딩으로 변경된 train_oh 배열이 아니라 train_seq를 사용할 수 있다
# 이 때문에 메모리를 훨씬 효율적으로 사용할 수 있다

# 원-핫 인코딩은 샘플 하나를 300차원으로 늘렸기 떄문에 (100, ) 크기의 샘플이
# (100, 300)으로 커졌다
# 이와 비슷하게 임베딩도 (100, ) 크기의 샘플을 (100, 20)과 같이 2차원 배열로 늘린다
# 원-핫 인코딩과는 달리 훨씬 작은 크기로도 단어를 잘 표현할 수 있다

# Embedding 클래스를 SimpleRNN 층 앞에 추가한 신경망 생성
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(300, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

# Embedding 클래스의 첫 번째 매개변수(300)는 어휘 사전의 크기이다
# 앞서 데이터셋에서 300개의 단어만 사용하도록 imdb.load_data(num_words=300)과 같이 
# 설정했기 때문에 이 매개변수의 값을 300으로 지정한다

# 두 번째 매개변수(16)는 임베딩 벡터의 크기다

# 세 번쨰 input_length 매개변수는 입력 시퀀스의 길이이다 
# 앞서 샘플의 길이를 100으로 맞추어 train_seq를 만들었다 따라서 이 값은 100이다

# 모델 구조 
model2.summary()
# Model: "sequential_3"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ embedding (Embedding)                │ ?                           │         4800    │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ simple_rnn_3 (SimpleRNN)             │ ?                           │     0 (unbuilt) │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_3 (Dense)                      │ ?                           │     0 (unbuilt) │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 0 (0.00 B)
#  Trainable params: 0 (0.00 B)
#  Non-trainable params: 0 (0.00 B)

# Embedding 클래스는 300개의 각 토큰을 크기가 16인 벡터로 변경하기 때문에 300 * 16 = 4800개의 모델 파라미터를 가진다
# SimpleRNN 층은 임베딩 벡터의 크기가 16이므로 8개의 뉴련과 곱하기 위해 필요한 가중치 16 * 8 = 128개를 가진다
# 또한 은닉 상태에 곱해지는 가중치 8 * 8 = 64개가 있다
# 마지막으로 8개의 절편이 있므므로 이 순환층에 있는 전체 모델 파라미터의 개수는 128 + 64 + 8 = 200개이다
# 마지막 Dense 층의 가중치 개수는 이전과 동일하게 9개이다

# 원-핫 인코딩보다 SimpleRNN에 주입되는 입력의 크기가 크게 줄었지만 임베딩 벡터는 단어를 잘 표현하는 
# 능력이 있기 때문에 훈련 결과는 이전보다 좋을 것이다

# 모델 훈련
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model2.fit(
    train_seq, train_target, 
    epochs=100, batch_size=64, 
    validation_data=(val_seq, val_target)
    callbacks=[checkpoint_cb, early_stopping_cb]
)
# Epoch 1/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 24ms/step - accuracy: 0.5125 - loss: 0.6950 - val_accuracy: 0.5808 - val_loss: 0.6814
# Epoch 2/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 30ms/step - accuracy: 0.6068 - loss: 0.6741 - val_accuracy: 0.6474 - val_loss: 0.6596
# Epoch 3/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 28ms/step - accuracy: 0.6544 - loss: 0.6537 - val_accuracy: 0.6808 - val_loss: 0.6396
# Epoch 4/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 23ms/step - accuracy: 0.6902 - loss: 0.6338 - val_accuracy: 0.6784 - val_loss: 0.6291
# Epoch 5/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 24ms/step - accuracy: 0.7067 - loss: 0.6142 - val_accuracy: 0.6682 - val_loss: 0.6252
# Epoch 6/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 29ms/step - accuracy: 0.7244 - loss: 0.5986 - val_accuracy: 0.7252 - val_loss: 0.5921
# Epoch 7/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 35ms/step - accuracy: 0.7331 - loss: 0.5824 - val_accuracy: 0.7366 - val_loss: 0.5747
# Epoch 8/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 8s 25ms/step - accuracy: 0.7404 - loss: 0.5694 - val_accuracy: 0.7270 - val_loss: 0.5704
# Epoch 9/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 24ms/step - accuracy: 0.7504 - loss: 0.5553 - val_accuracy: 0.7448 - val_loss: 0.5558
# Epoch 10/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 29ms/step - accuracy: 0.7507 - loss: 0.5463 - val_accuracy: 0.7434 - val_loss: 0.5475
# Epoch 11/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 31ms/step - accuracy: 0.7570 - loss: 0.5338 - val_accuracy: 0.7340 - val_loss: 0.5523
# Epoch 12/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 8s 24ms/step - accuracy: 0.7494 - loss: 0.5357 - val_accuracy: 0.7328 - val_loss: 0.5426
# Epoch 13/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 24ms/step - accuracy: 0.7612 - loss: 0.5215 - val_accuracy: 0.7474 - val_loss: 0.5317
# Epoch 14/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 12s 29ms/step - accuracy: 0.7585 - loss: 0.5190 - val_accuracy: 0.7368 - val_loss: 0.5411
# Epoch 15/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 30ms/step - accuracy: 0.7638 - loss: 0.5119 - val_accuracy: 0.7488 - val_loss: 0.5240
# Epoch 16/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.7633 - loss: 0.5099 - val_accuracy: 0.7452 - val_loss: 0.5226
# Epoch 17/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 30ms/step - accuracy: 0.7673 - loss: 0.5023 - val_accuracy: 0.7350 - val_loss: 0.5278
# Epoch 18/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 32ms/step - accuracy: 0.7648 - loss: 0.5014 - val_accuracy: 0.7466 - val_loss: 0.5214
# Epoch 19/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 8s 24ms/step - accuracy: 0.7683 - loss: 0.4992 - val_accuracy: 0.7538 - val_loss: 0.5181
# Epoch 20/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 11s 27ms/step - accuracy: 0.7675 - loss: 0.4990 - val_accuracy: 0.7540 - val_loss: 0.5159
# Epoch 21/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 30ms/step - accuracy: 0.7732 - loss: 0.4898 - val_accuracy: 0.7526 - val_loss: 0.5157
# Epoch 22/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 7s 24ms/step - accuracy: 0.7695 - loss: 0.4922 - val_accuracy: 0.7514 - val_loss: 0.5149
# Epoch 23/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 30ms/step - accuracy: 0.7675 - loss: 0.4947 - val_accuracy: 0.7506 - val_loss: 0.5191
# Epoch 24/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.7750 - loss: 0.4848 - val_accuracy: 0.7508 - val_loss: 0.5116
# Epoch 25/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step - accuracy: 0.7789 - loss: 0.4877 - val_accuracy: 0.7520 - val_loss: 0.5152
# Epoch 26/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 29ms/step - accuracy: 0.7712 - loss: 0.4878 - val_accuracy: 0.7558 - val_loss: 0.5106
# Epoch 27/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 8s 24ms/step - accuracy: 0.7691 - loss: 0.4903 - val_accuracy: 0.7480 - val_loss: 0.5189
# Epoch 28/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 24ms/step - accuracy: 0.7730 - loss: 0.4862 - val_accuracy: 0.7536 - val_loss: 0.5108
# Epoch 29/100
# 313/313 ━━━━━━━━━━━━━━━━━━━━ 9s 30ms/step - accuracy: 0.7746 - loss: 0.4817 - val_accuracy: 0.7550 - val_loss: 0.5112