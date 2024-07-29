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