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

# 함수형 API (functional API)
# 딥러닝에서는 좀 더 복잡한 모델이 많이 있다
# 예를 들어 입력이 2개일 수도 있고 출력이 2개일 수도 있다
# 이런 경우 Sequential 클래스를 사용하기 어렵다
# 대신 함수형 API를 사용한다

# 함수형 API는 케리사의 Model 클래스를 사용하여 모델을 만든다

# Dense층 2개로 이루어진 완전 연결 신경망을 함수형 API로 구현
# 2개의 Dense 층 객체 생성
dense1 = keras.layers.Dense(100, activation='sigmoid')
dense2 = keras.layers.Dense(10, activation='softmax')

# 이 객체를 Sequential 클래스 객체의 add 메서드에 전달할 수 있다
# 하지만 다음과 같이 함수처럼 호출할 수도 있다
hidden = dense1(inputs)

# 파이썬의 모든 객체는 호출 가능하다
# 케라스의 층은 객체를 함수처럼 호출했을 때 적절히 동작할 수 있도록 미리 준비해 놓았다
# 해당 코드를 실행하면 입력값 inputs를 Dense 층을 통과시킨 후 출력값 hidden을 만든다

# 두 번째 층 호출 - 첫 번째 층의 출력을 입력으로 사용
outputs = dense2(hidden)

# inputs와 outputs을 Model 클래스로 연결
model = keras.Model(inputs, outputs)

# 입력에서 출력까지 층을 호출한 결과를 계속 이어주고 Model 클래스에 입력과 최종 출력을 지정한다
# Sequential 클래스는 InputLayer 클래스를 자동으로 추가하고 호출해 주지만 Model 클래스에서는
# 수동으로 만들어서 호출해야 한다 
# inputs가 InputLayer 클래스의 출력값이 되어야 한다

# 케라스는 InputLayer 클래스 객체를 쉽게 다룰 수 있도록 Input 함수를 별도로 제공
# 입력의 크기를 지정하는 shape 매개변수와 함께 이 함수를 호출하면 InputLayer 클래스 객체를
# 만들어 출력을 반환
inputs = keras.Input(shape=(784,))

# 이렇게 모델을 만들게 되면 중간에 다양한 형태로 층을 연결할 수 있다

# model 객체의 predict 메서드를 호출하면 입력부터 마지막 층까지 모든 계산을 수행한 후
# 최종 출력을 반환
# 필요한 데이터가 첫 번쨰 Conv2D층이 출력한 특성 맵이다
# 첫 번째 층의 출력은 Conv2D 객체의 output 속성에서 얻을 수 있다
# model.layers[0].output처럼 참초 가능

# model 객체의 입력은 input 속성으로 입력을 참조할 수 있다
print(model.input)
# KerasTensor(type_spec=TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32, name='conv2d_input'), name='conv2d_input', description="created by layer 'conv2d_input'")

# model.input과 model.layers[0].output을 연결하는 모델 생성
conv_acti = keras.Model(model.input, model.layers[0].output)

# model 객체의 predict 메서드를 호출하면 최종 출력층의 확률을 반환한다
# conv_acti의 predict 메서드를 호출하면 첫 번째 Conv2D의 출력을 반환할 것이다

# 케라스로 패션 MNIST 데이터셋을 읽은 후 훈련 세트에 있는 첫 번째 샘플 출력
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap='gray_r')
plt.show()

# 이 샘플을 conv_acti 모델에 주입하여 Conv2D 층이 만드는 특성 맵 출력
# predict 메서드는 항상 입력의 첫 번째 차원이 배치 차원일 것으로 기대하므로 
# 샘플 전달 전 차원을 유지해야 한다 작업 후 전처리
inputs = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv_acti.predict(inputs)

# predict 메서드가 출력한 크기 확인
print(feature_maps.shape)
# (1, 28, 28, 32)
# 세임 패딩과 32개의 필터를 사용한 합성곱 층의 출력이므로 (28, 28, 32)이다

# 맷플롯립의 imshow 함수로 특성 맵 출력
# 총 32개의 특성 맵을 4개의 행으로 나누어 출력
fig, axs = plt.subplots(4, 8, figsize=(15, 8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i * 8 + j])
        axs[i, j].axis('off')
plt.show()