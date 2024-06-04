fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# 넘파이 임포트
import numpy as np

# 넘파이의 column_stack() 함수는 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결
# 연결할 리스트는 파이썬 튜플로 전달
np.column_stack(([1, 2, 3], [4, 5, 6]))

# 아래처럼 출력됨
# arry([[1, 4],
#       [2, 5],
#       [3, 6]])

# 물고기의 길이와 무게를 column_stack을 사용하여 손쉽게 병합
fish_data = np.column_stack((fish_length, fish_weight))

# 5개의 데이터 출력하여 검증
print(fish_data[:5])

# 타깃 데이터를 넘파이를 사용하여 손쉽게 작성
# np.ones(), np.zeros() 함수를 사용하면 각각 원하는 개수의 1과 0을 채운 배열을 만들어 준다
print(np.ones(5))

# 이 함수를 사용해 1이 35개인 배열과 0이 14개인 배열을 간단히 만들 수 있다
# 그 후 두 배열을 그대로 연결하면 된다
# 첫 번째 차원을 따라 배열을 연결하는 np.concatenate()함수를 사용
# 연결할 리스트나 배열을 튜플로 전달해야 한다
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

print(fish_target)

# 데이터가 큰 경우 파이썬 리스트로 작업하는 것은 비효율적 
# 넘파이 배열은 핵심 부분이 C, C++과 같은 저수준 언어로 개발되어서 빠르고 데이터 과학 분야에 알맞게 최적화되어 있음

# 사이킷런은 머신러닝 모델을 위한 알고리즘뿐만 아니라 다양한 유틸리티 도구도 제공
# 사이킷런의 train_test_split() 함수를 사용하면 전달되는 리스트나 배열을 비율에 맞게 훈련 세트와 테스트 세트로 나누어 주며 나누기 전에 알아서 섞는다

# train_test_split() 함수는 사이킷런의 model_selection 모듈 아래에 있다
from sklearn.model_selection import train_test_split

# 나누고 싶은 리스트나 배열을 원하는 만큼 전달하면 된다
# train_test_split() 함수에는 자체적으로 랜덤 시드를 지정할 수 있는 random_state 매개변수가 있다
# fish_data와 fish_target 2개의 배열을 전달했으므로 2개씩 나뉘어 입력 데이터, 타깃 데이터 총 4개의 배열이 반환된다
# 이 함수는 기본적으로 25%를 테스트 세트로 떼어낸다
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)

# 입력 데이터 검증
# 2차원 배열
print(train_input.shape, test_input.shape)
# 타깃 데이터 검증
# 1차원 배열 
# 원소가 하나인 튜플 반환
print(train_target.shape, test_target.shape)

# 데이터가 잘 섞였는지 검증
# 해당 데이터는 샘플링 편향이 일어나 빙어가 적다
print(test_target)

# train_test_split() 함수에 stratify 매개변수로 타깃 데이터를 전달하면 클래스 비율에 맞게 데이터를 나눈다
# 훈련 데이터가 작거나 특정 클래스의 샘플 개수가 적을 때 유용하다

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42
)

# 샘플링 편향이 사라졌는지 검증
print(test_target)
