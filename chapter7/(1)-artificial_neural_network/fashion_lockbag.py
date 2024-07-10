# MNIST
# 머신러닝과 딥러닝을 처음 배울 때 많이 사용하는 데이터셋이 있는데
# 머신러닝은 붓꽃 데이터셋을 많이 쓰고 딥러닝은 MNIST 데이터셋이 유명하다
# MNIST는 손으로 쓴 0~9까지의 숫자로 이루어져 있다

# 텐서플로를 사용해 데이터 불러오기
# 텐서플로의 케라스 패키지를 임포트하고 패션 MNIST 데이터를 다운로드
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

# keras.datasets.fashion_mnist 모듈 아래 load_data() 함수는 훈련 데이터와 테스트 데이터를 나누어 반환
# 이 데이터는 각각 입력과 타깃의 쌍으로 구성

# 전달받은 데이터의 크기 확인
print(train_input.shape, train_target.shape)
# (60000, 28, 28) (60000,)

# 훈련 데이터는 60000개의 이미지로 이루어져 있고 각 이미지는 28 X 28 크기다
# 타깃도 60000개의 원소가 있는 1차원 배열이다

# 테스트 세트의 크기 확인
print(test_input.shape, test_target.shape)
# (10000, 28, 28) (10000,)

# 훈련 데이터에서 몇 개의 샘플 출력
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10, 10))

for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')

plt.show()

# 파이썬의 리스트 내포를 사용해서 처음 10개 샘플의 타깃값을 리스트로 만든 후 출력
print([train_target[i] for i in range(10)])
# [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]

# 패션 MNIST의 타깃은 0~9까지의 숫자 레이블로 구성
# 패션 MNIST에 포함된 10개 레이블의 의미는 다음과 같다
# 0      1   2      3     4    5    6   7        8   9
# 티셔츠 바지 스웨터 드레스 코트 샌달 셔츠 스니커즈 가방 앵클부츠

# 넘파이 unique() 함수로 레이블 당 샘플 개수를 확인
import numpy as np 
print(np.unique(train_target, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))

# 이 훈련 샘플은 60000개나 되기 때문에 전체 데이터를 한꺼번에 사용하여 모델을 훈련하는 것보다 샘플을 
# 하나씩 꺼내서 모델을 훈련하는 방법이 더 효율적이다

# 이런 상황에 잘 맞는 방법이 확률적 경사 하강법이다

# SGDClassifier를 사용할 때 표준화 전처리된 데이터를 사용했다
# 확률적 경사 하강법은 여러 특성 중 기울기가 가장 가파른 방향을 따라 이동한다
# 만약 특성마다 값의 범위가 많이 다르면 올바르게 손실 함수의 경사를 내려올 수 없다

# 패션 MNIST의 경우 각 픽셀은 0~255 사이의 정숫값을 가진다
# 이런 이미지의 경우 보통 255로 나누어 0~1 사이의 값으로 정규화한다
# 이는 표준화는 아니지만 양수 값으로 이루어진 이미지를 전처리할 때 널리 사용하는 방법이다

# SGDClassifier는 2차원 입력을 다루지 못한다
# reshape() 메서드를 사용해 2차원 배열인 각 샘플을 1차원 배열로 펼친다
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28 * 28)

# reshape() 메서드의 두 번째 매개변수를 28 X 28 이미지 크기에 맞게 지정하면 
# 첫 번째 차원(샘플 개수)은 변하지 않고 원본 데이터의 두 번째,세 번째 차원이 1차원으로 합쳐진다

# 변환된 train_scaled의 크기 확인
print(train_scaled.shape)
# (60000, 784)

# 784개의 픽셀로 이루어진 60000개의 샘플
# SGDClassifier 클래스와 cross_validate 함수를 사용해 교차 검증으로 성능 확인
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)

print(np.mean(scores['test_score']))
# 0.8196000000000001

# 로지스틱 회귀 공식을 패션 MNIST 데이터에 맞게 변형하면 
# 촐 784개의 특성이 있으므로 아주 긴 식이 만들어진다
# 가중치 개수도 많아진다

# 인공 신경망
# 가장 기본적인 인공 신경망은 확률적 경사 하강법을 사용하는 로지스틱 회귀와 같다

# 출력층(output layer)
# 클래스를 계산하고 이를 바탕으로 클래스를 예측하기 때문에 신경망의 최종 값을 
# 만든다는 의미에서 출력층이라고 부른다

# 뉴련(neuron)
# 인공 신경망에서는 z 값을 계산하는 단위를 뉴런이라고 부른다
# 뉴런에서 일어나는 일은 선형 계산이 전부다
# 이제는 뉴런이란 표현 대신 유닛(unit)이라고 부르는 사람이 더 많아지고 있다

# 입력층(input layer)
# 입력층은 특성 자체이고 특별한 계산을 수행하지 않는다
# 많은 사람이 입력층이라고 부른다

# 매컬러-피츠 뉴런
# 1943년 워런 매컬러와 월터 피츠가 제안한 뉴런 모델
# 이런 인공 뉴런은 생물학적 뉴런에서 영감을 얻어 만들어졌다
# 인공 뉴런은 생물학적 뉴런의 모양을 본뜬 수학 모델에 불과하다
# 인공 신경망은 정말 우리의 뇌에 있는 뉴런과 같지 않다
# 인공 신경망은 기존의 머신러닝 알고리즘이 잘 해결하지 못했던 문제에서 높은 성능을 발휘하는
# 새로운 종류의 머신러닝 알고리즘일 뿐이다

# 딥러닝
# 인공 신경망과 거의 동의어로 사용되는 경우가 많다
# 혹은 심층 신경망(depp neural network, DNN)을 딥러닝이라고 부른다
# 심층 신경망은 여러 개의 층을 가진 인공 신경망이다


