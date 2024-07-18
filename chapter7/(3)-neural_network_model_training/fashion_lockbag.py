# 케라스의 fit() 메서드는 History 클래스 객체를 반환한다
# History 객체에는 훈련 과정에서 계산한 지표, 즉 손실과 정확도 값이 저장되어 있다

# 패션 MNIST 데이터셋을 적재하고 훈련 세트와 검증 세트로 나눔
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42
)

# 모델을 만드는 간단한 함수 정의
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(100, activation='relu'))

    if a_layer:
        model.add(a_layer)
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# 함수로 모델 생성
model = model_fn()
model.summary()
# Model: "sequential_12"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  flatten_5 (Flatten)         (None, 784)               0         
                                                                 
#  dense_22 (Dense)            (None, 100)               78500     
                                                                 
#  dense_23 (Dense)            (None, 10)                1010      
                                                                 
# =================================================================
# Total params: 79510 (310.59 KB)
# Trainable params: 79510 (310.59 KB)
# Non-trainable params: 0 (0.00 Byte)
# _________________________________________________________________

# History 객체 담기
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(train_scaled, train_target, epochs=5, verbose=0)

# verbosse 매개변수는 훈련 과정 출력을 조절
# 기본값은 1로 에포크마다 진행 막대와 함께 손실 등의 지표가 풀력
# 2로 바꾸면 진행 막대를 뺴고 출력 
# 0은 훈련 과정을 나타내지 않는다

# history 객체에는 훈련 측정값이 담겨 있는 history 딕셔너리가 들어 있다
print(history.history.keys())
# dict_keys(['loss', 'accuracy'])
# 손실과 정확도가 포함되어있다

# 케라스는 기본적으로 에포크마다 손실을 계산한다
# 정확도는 compile() 메서드에서 metrics 매개변수에 accuracy를 추가했기 떄문에 history 속성에 포함되어 있다
# history 속성에 포함된 손실과 정확도는 에포크마다 계산한 값이 순서대로 나열된 단순 리스트이다

# 맷플롯립으로 그래프 출력
# 손실
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 정확도
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()