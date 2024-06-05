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


# 넘파이와 train_test_split을 사용하여 만든 데이터 셋으로 훈련
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
# 훈련
kn.fit(train_input, train_target)
# 검증
kn.score(test_input, test_target)

# 도미 데이터를 넣고 확인
# 결과는 빙어로 나온다
print(kn.predict([[25, 150]]))

# 산점도로 도미 데이터 출력
import matplotlib.pyplot as plt

# 길이와 무게로 x,y축 지정
plt.scatter(train_input[:, 0], train_input[: ,1])
# 이상 데이터를 표시하기 위해 marker 매개변수로 모양 설정
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 이상 데이터 샘플은 도미 데이터 군집에 더 가깝게 그래프가 생성된다
# 그럼에도 빙어로 출력된다

# k-최근접 이웃은 주변의 샘플 중에서 다수인 클래스를 예측으로 사용한다
# 주변 샘플 검증을 위해 KNeighborsClassifier에서 kenighbors() 메서드를 사용해 주변 샘플 추출

# kenighbors()
# 주어진 샘플에서 가장 가까운 이웃을 찾아주는 메서드
# 이웃까지의 거리와 이웃 샘플의 인덱스를 반환 (배열)
# KNeighborsClassifier에서 사용하는 이웃 샘플은 기본값이 5다

distances, indexes = kn.kneighbors([[25, 150]])

# 산점도로 이웃 샘플을 알아보기 쉽게 렌더링
# 전체 데이터
plt.scatter(train_input[:, 0], train_input[:, 1])
# 이상 데이터 표시
plt.scatter(25, 150, marker='^')
# 정답 데이터 유추로 쓰인 이웃 데이터들 렌더링 및 표시
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 결과를 보면 가까운 이웃에 도미가 하나고 나머지 4개가 빙어로 나온다

# 이웃 데이터 출력
print(train_input[indexes])
# 이웃 타깃 데이터 출력
print(train_target[indexes])

# 문제 해결 실마리를 찾기 위해 distances(이웃 샘플간의 거리) 배열을 출력 
print(distances)

# 산점도를 보면 거리가 92로 나온 데이터보다 130으로 나온 데이터의 거리가 몇 배는 되어 보이지만 거리는 가깝게 나왔다
# x 축은 범위가 좁고(10 ~ 40) y축은 범위가 넓다(0 ~ 1000) 
# 따라서 y 축으로 조금만 멀어져도 거리가 아주 큰 값으로 계산된다
# 이 때문에 산점도 오른쪽 위의 도미 샘플이 이웃으로 선택되지 못했다

# 이를 눈으로 명확히 확인하기 위해 x축의 범위를 동일하게 0 ~ 1000으로 맞춰보자
# x축 범위 지정 xlim() 함수 사용
# y축 범위 지정 ylim() 함수 사용
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim((0, 1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 두 특성의 값이 놓은 범위가 매우 다르다 이를 두 특성의 스케일이 다르다고 말한다 
# 특성 간 스케일이 다른 일은 매우 흔한다 
# 어떤 사람이 방의 넓이를 재는데 세로는 cm로 가로는 inch로 쟀다면 정사각형인 방도 직사각형처럼 보일 것이다
# 데이터를 표현하는 기준이 다르면 알고리즘은 올바르게 예측할 수 없다 
# 알고리즘이 거리 기반일 경우 더욱 영향을 받는다 (k-최근접 이웃 포함)
# 특성값을 일정한 기준으로 맞춰 주어야 한다 이런 작업을 데이터 전처리라고 부른다

# 가장 널리 사용하는 전처리 방법 중 하나는 표준점수(z 점수)이다
# 표준점수는 각 특성값이 평균에서 표준편차의 몇 배만큼 떨어져 있는지를 나타낸다 
# 이를 통해 실제 특성값의 크기와 상관없이 동일한 조건으로 비교할 수 있다

# 분산은 데이터에서 평균을 뺀 값을 모두 제곱한 다음 평균을 내어 구한다 
# 표준편차는 분산의 제곱근으로 데이터가 분산된 정도를 나타낸다
# 표준점수는 각 데이터가 원점에서 몇 표준편차만큼 떨어져 있는지를 나타내는 값이다

# 평균을 빼고 표준편차를 나누어 계산 
# 넘파이 함수를 이용하여 계산

# np.mean() = 평균을 계산
mean = np.mean(train_input, axis=0)
# np.std() = 표준편차를 계산
std = np.std(train_input, axis=0)
# 특성마가 값의 스케일이 다르므로 평균과 표준편차는 각 특성별로 계산해야 한다
# 이를 위해 axis = 0 으로 지정 이렇게 하면 행을 따라 각 열의 통계 값을 계산

# 평균과 표준편차 출력
print(mean, std)

# 각 특성의 평균과 표준편차가 구해졌으므로 원본 데이터에서 평균을 빼고 표준편차로 나누어 표준점수로 변환
# 넘파이가 train_input 의 모든 행에서 mean에 있는 두 평균값을 빼준다
# 그 다음 std에 있는 두 표준편차를 다시 모든 행에 적용한다
# 이런 넘파이 기능을 브로드캐스팅이라고 부른다
# 브로드 캐스팅은 넘파이 배열 사이에서 일어난다 즉 train_input, mean, std 모두 넘파이 배열이다
train_scaled = (train_input - mean) / std