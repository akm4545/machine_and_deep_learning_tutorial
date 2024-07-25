# 텐서플로를 사용하면 합성곱, 패딩, 풀링 크기를 직접 계산할 필요가 없다
# 복잡한 계산은 케라스 API에 모두 위임하고 사용자는 직관적으로 
# 신경망을 설계할 수 있다

# 패션 MNIST 데이터를 불러오고 전처리
# 데이터 스케일을 0~255 사이에서 0~1 사이로 바꾸고 훈랸 세트와 검증 세트로 나눔
# 완전 연결 신경망에서는 입력 이미지를 밀집층에 연결하기 위해 일렬로 펼쳐야 했다
# 합성곱 신경망은 2차원 이미지를 그대로 사용하기 때문에 일렬로 펼치지 않는다

# 입력 이미지는 항상 깊이(채널) 차원이 있어야 한다
# 흑백 이미지의 경우 채널 차원이 없는 2차원 배열이라 Conv2D 층을 사용하기 위해
# 마지막에 이 채널 차원을 추가해야 한다

# 넘파이 reshape() 메서드를 사용해 전체 배열 차원을 그대로 유지하면서 마지막에 차원 추가
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 케라스의 Sequential 클래스를 사용해 구조 정의
# 합성곱 층인 Conv2D 추가
# 이 클래스는 다른 층 클래스와 마찬가지로 keras.layers 패키지 아래에 있다
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))

# 해당 합성곱은 32개의 필터를 사용한다
# 커널의 크기는 (3, 3)이고 렐루 활성화 함수와 세임 패딩을 사용
# 신경망 모델의 첫 번째 층에서 입력의 차원을 지정해 주어야 한다
# 패션 MNIST 이미지를 (28, 28)에서 (28, 28, 1)로 변경했으므로 input_shape 매개변수를 이 값으로 지정

# 풀링 층은 keras.layers 패키지 아래에 MaxPooling2D(최대 풀링), AveragePooling2D(평균 풀링)
# 클래스로 제공
# Conv2D 클래스의 kernel_size 처럼 가로세로 크기가 같으면 정수 하나로 지정할 수 있다
model.add(keras.layers.MaxPooling2D(2))

# 세임 패딩을 적용했기 떄문에 합성곱 층에서 출력된 특성 맵의 가로세로 크기는 입력과 동일하다
# 그 다음 (2, 2) 풀링을 적용했으므로 특성 맵의 크기는 절반으로 줄어든다
# 합성곱 층에서 32개의 필터를 사용했기 때뭉네 이 특성 맵의 깊이는 32가 된다
# 따라서 최대 풀링을 통과한 특성 맵의 크기는 (14, 14, 32)가 될 것이다

# 두 번째 합성곱-풀링 층 추가
# 필터 개수 64개
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

# (밀집) 출력층에서 확률 계산을 위해 특성 맵을 일렬로 펼쳐야 한다
# Flatten클래스 -> Dense은닉층 -> Dropout 층 (과대적합 방지) -> Dense 출력층의 순서대로 구상
model.add(keras.layers.Flatten())
# 100개의 뉴런을 사용하고 활성화 함수는 렐루 함수 사용
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
# 10개의 클래스를 분류하는 다중 분류 문제이므로 마지막 층의 활성화 함수는 소프트맥스 사용
model.add(keras.layers.Dense(10, activation='softmax'))

# 모델 구조 출력
model.summary()
# Model: "sequential_3"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d_4 (Conv2D)           (None, 28, 28, 32)        320       
                                                                 
#  max_pooling2d_3 (MaxPoolin  (None, 14, 14, 32)        0         
#  g2D)                                                            
                                                                 
#  conv2d_5 (Conv2D)           (None, 14, 14, 64)        18496     
                                                                 
#  max_pooling2d_4 (MaxPoolin  (None, 7, 7, 64)          0         
#  g2D)                                                            
                                                                 
#  flatten_1 (Flatten)         (None, 3136)              0         
                                                                 
#  dense_2 (Dense)             (None, 100)               313700    
                                                                 
#  dropout_1 (Dropout)         (None, 100)               0         
                                                                 
#  dense_3 (Dense)             (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 333526 (1.27 MB)
# Trainable params: 333526 (1.27 MB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# 층의 구성을 그림으로 표현해 주는 plot_model() 함수도 있다
# 이 함수는 keras.utils 패키지에 있다
keras.utils.plot_model(model)

# plot_model() 함수의 show_shapes 매개변수를 True로 설정하면 이 그림에 입력과 출력의
# 크기를 표시해 준다
# to_file 매개변수에 파일 이름을 지정하면 출력한 이미지를 파일로 저장한다
# dpi 매개변수로 해상도를 지정할 수도 있다
keras.utils.plot_model(model, show_shapes=True)