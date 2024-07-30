# 가중치 시각화
# 합성곱 층은 여러 개의 필터를 사용해 이미지에서 특징을 학습한다
# 각 필터는 커널이라 불리는 가중치와 절편을 가지고 있다
# 일반적으로 절편은 시각적으로 의미가 있지 않다
# 가중치는 입력 이미지의 2차원 영역에 적용되어 어떤 특징을 크게 두드러지게 표현하는 역할을 한다

# 전에 만든 모델을 읽기
from tensorflow import keras
model = keras.models.load_model('best-cnn-model.h5')

# 케라스 모델에 추가한 층은 layers 속성에 저장되어 있다
# 층 출력
model.layers
# [<keras.src.layers.convolutional.conv2d.Conv2D at 0x78b958c36740>,
#  <keras.src.layers.pooling.max_pooling2d.MaxPooling2D at 0x78b958c378b0>,
#  <keras.src.layers.convolutional.conv2d.Conv2D at 0x78b958c37f70>,
#  <keras.src.layers.pooling.max_pooling2d.MaxPooling2D at 0x78b958c373d0>,
#  <keras.src.layers.reshaping.flatten.Flatten at 0x78b958a8ca00>,
#  <keras.src.layers.core.dense.Dense at 0x78b958c37760>,
#  <keras.src.layers.regularization.dropout.Dropout at 0x78b958c36c20>,
#  <keras.src.layers.core.dense.Dense at 0x78b956d369b0>]

# 첫 번째 합성곱 층의 가중치 출력
# 가중치와 절편은 층의 weights 속성에 저장되어 있다
# weights는 파이썬 리스트이다

# layers 속성의 첫 번째 원소를 선택해 weights의 첫 번째 원소(가중치)와 두 번째 원소(절편)의 크기를 출력
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape)
# (3, 3, 1, 32) (32,)

# 커널 크기가 (3, 3)이고 입력의 킾이가 1이므로 실제 커널 크기는 (3, 3, 1)이다
# 필터 개수가 32개이므로 weights의 첫 번째 원소인 가중치의 크기는 (3, 3, 1, 32)가 되었다
# weights의 두 번째 원소는 절편의 개수를 나타낸다 
# 필터마다 1개의 절편이 있으므로 (32, )크기가 된다

# weights 속성은 텐서플로의 다차원 배열인 Tensor 클래스의 객체이다

# numpy로 넘파이 배열로 변환 후 가중치 배열의 평균과 표준편차를 mean 메서드와
# std 메서드로 계산
conv_weights = conv.weights[0].numpy()
print(conv_weights.mean(), conv_weights.std())
# -0.029794 0.28825074

# 이 가중치의 평균값은 0에 가깝고 표준편차는 0,28정도이다

# 출력한 가중치의 히스토그램 출력
import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# 맷플롯립의 hist() 함수에는 히스토그램을 그리기 위해 1차원 배열로 전달해야 한다
# 이를 위해 넘파이 reshape 메서드로 conv_weights 배열을 1개의 열이 있는 배열로 변환헀다
# 히스토그램을 보면 0을 중심으로 종 모양 분포를 띠고 있다

# 32개의 커널을 16개씩 두 줄에 출력
# 맷플롯립의 subplots 함수를 사용해 32개의 그래프 영역을 만들고 순서대로 커널 출력
fig, axs = plt.subplots(2, 16, figsize=(15, 2))

for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:,:,0,i * 16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# 배열의 마지막 차원을 순회하면서 0부터 i*16+j 번째까지의 가중치 값을 차례로 출력
# i = 행 인덱스, j = 열 인덱스로 각각 0~1, 0~15까지의 범위를 가진다
# 따라서 conv_weights[:,:,0,0]에서 conv_weights[:,:,0,31]까지 출력한다

# 결과 그래프를 보면 이 가중치 값이 무작위로 나열된 것이 아닌 어떤 패턴을 볼 수 있다

# imshow 함수는 배열에 있는 최댓값과 최솟값을 사용해 픽셀의 강도를 표현한다
# 어떤 값이든지 그 배열의 최댓값이면 가장 밝은 노란 색으로 그린다
# 만약 두 배열을 imshow 함수로 비교하려면 이런 동작은 바람직하지 않다
# 어떤 절댓값으로 기준을 정해서 픽셀의 강도를 나타내야 비교하기 편하다
# vmin과 vmax로 맷플롯립의 컬러맵으로 표현할 범위를 지정할 수 있다

# 훈련하지 않은 빈 합성곱 신경망 만들기
# Sequential 클래스로 모델을 만들고 Conv2D 층을 하나 추가
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))

# 모델의 첫 번째 층(Conv2D)의 가중치를 no_training_conv 변수에 저장
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)
# (3, 3, 1, 32)

# 가중치의 평균과 표춘편차 출력
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())
# 0.0014820709 0.08248861

# 평균은 0에 가깝지만 표준편차는 이전과 달리 매우 작다

# 가중치 배열을 히스토그램으로 출력
plt.hist(no_training_weights.reshape(-1, 1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# 출력 결과 대부분의 가중치가 -0.15~0.15 사이에 있고 비교적 고른 분포를 보인다
# 이런 이유는 텐서플로가 신경망의 가중치를 처음 초기화할 때 균등 분포에서 랜덤하게 값을 선택하기 때문이다

# 가중치 값을 맷플롯립의 imshow 함수를 사용해 그림으로 출력
# 학습 가중치와 비교하기 위해 동일하게 vmin과 vmax를 -0.5와 0.5로 설정

fig, axs = plt.subplots(2, 16, figsize=(15, 2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:,:,0,i * 16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# 가중치가 밋밋하게 초기화되었다
# 합성곱 신경망의 학습을 시각화하는 두 번째 방법은 합성곱 층에서 출력된 특성 맵을 그려 보는 것이다
# 이를 통해 입력 이미지를 신경망 층이 어떻게 바로지는지 엿볼 수 있다

# 함수형 API