# 테스트 세트 물고기 데이터
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 각 생선의 길이와 무게를 담은 2차원 데이터 생성
# 하나의 물고기 데이터를 샘플이라고 부른다
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
# 정답 데이터 생성
fish_target = [1] * 35 + [0] * 14

# 사이킷런의 KNeighborsClassifier 클래스 임포트
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()

# 처음 35개는 훈련 세트 나머지 14개는 테스트 세트로 사용

# 슬라이싱 연산으로 배열 추출 
# :35 = 0 ~ 34번째 인덱스까지 추출
# 훈련 세트로 입력값 중 0 부터 34번째 인덱스까지 사용
train_input = fish_data[:35]

# 훈련 세트로 타깃값 중 0부터 34번째 인덱스까지 사용
train_target = fish_target[:35]

# 테스트 세트로 입려값 중 35번째부터 마지막 인덱스까지 사용
test_input = fish_data[35:]

# 테스트 세트로 타깃값 중 35번째부터 마지막 인덱스까지 사용
test_target = fish_target[35:]

# 훈련
kn.fit(train_input, train_target)
# 평가
# 도미 데이터로만 훈련시켰기 때문에 0.0이 나온다
kn.score(test_input, test_target)

# 넘파이 파이브러리 임포트
import numpy as np 

# 파이썬 리스트를 넘파이 배열로 바꾸기 
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

print(input_arr)

# 넘파이 배열 객체는 배열의 크기를 알려주는 shape 속성을 제공
# (샘플 수, 특성 수) 출력
print(input_arr.shape)

# 배열을 섞은 후 나누는 방식 대신에 무작위 샘플을 고르는 방법 사용
# 주의할 점은 input_arr와 target_arr에서 같은 위치는 함께 선택되어야 한다는 점이다
# 예를 들어 input_arr의 두 번째 값은 훈련 세트로 가고 target_arr의 두 번째 값은 테스트 세트로 가면 안된다 학습 데이터와 정답 데이터의 불일치가 일어나기 때문
# 타깃이 샘플과 함께 이동하지 않으면 올바르게 훈련될 수 없다

# 인덱스를 섞은 다음 input_arr와 target_arr에서 샘플을 선택하면 무작위로 훈련 세트를 나누는 셈이 된다

# 동일한 실습 결과를 얻기 위해 랜덤 시드를 설정 
np.random.seed(42)
# arange()함수를 사용하여 1씩 증가하는 인덱스를 간단히 만들 수 있다
# 정수 N을 전달하면 0에서부터 N-1까지 1씩 증가하는 배열을 만든다
index = np.arange(49)
# shuffle() 주어진 배열을 무작위로 섞는다 
np.random.shuffle(index)

print(index)

# 렌덤하게 섞인 인덱스를 사용해 전체 데이터를 훈련 세트와 테스트 세트로 나눈다

# 배열 인덱싱
# 첫 번째와 네 번째 샘플 선택하여 출력 코드
# print(input_arr[[1, 3]])

# 렌덤하게 섞은 index 배열을 배열 인덱싱을 사용하여 넘파이 배열 생성
# 훈련 세트 생성
train_input = input_arr[index[:35]]
train_target = input_arr[index[:35]]

# 값 검증
# 섞은 인덱스의 첫번째가 13이므로 
# train_input의 0번째 인덱스로 input_arr[13]번째 데이터가 들어갔다
print(input_arr[13], train_input[0])

# 나머지 14개를 테스트 세트로 생성
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 훈련 세트와 테스트 세트에 도미와 빙어가 잘 섞여 있는지 산점도로 검증
import matplotlib.pyplot as plt

# 2차원 배열은 행과 열 인덱스를 콤마(,)로 나누어 지정 
# 슬라이싱 연산자로 처음부터 마지막 원소까지 모두 선택하는 경우 시작과 종료 인덱스 생략 가능
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(test_input[:,0], test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
