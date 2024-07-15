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