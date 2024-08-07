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