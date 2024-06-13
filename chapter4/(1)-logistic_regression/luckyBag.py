# 랜덤박스의 생성의 확률 계산 

# 데이터 준비
# 판다스로 csv 데이터를 읽어 데이터 프레임으로 변환 후 출력
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
# 처음 5개 행 출력
fish.head()

# 데이터프레임은 판다스에서 제공하는 2차원 표 형식의 주요 데이터 구조
# 넘파이 배열과 비슷하게 열과 행으로 이루어져 있다
# 데이터프레임은 통계와 그래프를 위한 메서드를 풍부하게 제공한다 
# 데이터 프레임은 넘파이로 상호 변환이 쉽고 사이킷런과도 잘 호환된다

# 어떤 종류의 생선이 있는지 판다스의 unique() 함수를 이용해 Species 열에서 고유값 추출
print(pd.unique(fish['Species']))

# Species 열을 타깃으로 만들고 나머지 5개 열은 입력 데이터로 사용
# 데이터 프레임에서 열을 선택하는 방법은 원하는 열을 리스트로 나열하면 된다
# to_numpy() 메서드로 넘파이 배열로 변환
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()

print(fish_input[:5])

# 타깃 데이터 추출
# fish[['Species']]와 같이 사용하면 2차원 배열이 되기 때문에 조심해야 한다
fish_target = fish['Species'].to_numpy()

# 훈련세트, 테스트세트 생성
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42
)

# 훈련세트, 테스트세트 표준화 전처리
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# k-최근접 이웃 분류기의 확률 예측
# KNeighborsClassifier 클래스 객체를 사용하여 훈련
from sklearn.neighbors import KNeighborsClassifier

kn = KneighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
# 0.89
print(kn.score(test_scaled, test_target))
# 0.85

# 현재 타깃 데이터에 2개 이상의 클래스가 들어있는데 이런 문제를 다중 분류라고 부른다
# 2장에서는 타깃 값을 1과 0으로 만들었지만 문자열로 된 타깃값을 그대로 사용할 수 있다
# 타깃값을 그대로 사이킷런 모델에 전달하면 순서가 자동으로 알파벳 순으로 매겨진다
# 따라서 pd.unique(fish['Species'])로 출력했던 순서와 다르다
# KNeighborsClassifier에서 정렬된 타깃값은 classes_ 속성에 저장
print(kn.classes_)

# predict() 메서드는 타깃값으로 예측을 출력한다 
print(kn.predict(test_scaled[:5]))
# ['Perch', 'Smelt', 'Pike', 'Perch', 'Perch']

# 사이킷런 분류 모델은 predict_proba() 메서드로 클래스별 확률값을 반환
# 넘파이 round() 함수는 기본으로 소수점 첫째 자리에서 반올림을 하는데 decimals 매개변수로 유지할 소수점 아래 자릿수를 지정 가능
import numpy as np

proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))
#  샘플 -> [ 첫 번째 클래스(Bream)에 대한 확률. 두 번째 클래스(Parkki)에 대한 확률 ...]
# 순서는 print(kn.classes_)에서 출력되는 순으로 지정된다
# [[0.     0.     1.     0.     0.     0.     0.    ]
#  [0.     0.     0.     0.     0.     1.     0.    ]
#  [0.     0.     0.     1.     0.     0.     0.    ]
#  [0.     0.     0.6667 0.     0.3333 0.     0.    ]
#  [0.     0.     0.6667 0.     0.3333 0.     0.    ]]

# 모델이 계산한 확률이 가장 가까운 이웃의 비율 검증
# kneighbors() 메서드의 입력은 2차원 배열이여야 한다
# 넘파이의 슬라이싱 연산자는 하나의 샘플만 선택해도 항상 2차원 배열이 만들어진다

# 4번째 샘플의 이웃 출력
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
# [['Roach' 'Perch' 'Perch']]

# Roach가 1개이고 Perch가 2개다
# 따라서 세 번째 클래스에 대한 확률은 2/3 = 0.6667 이고 다섯번째 클래스에 대한 확률은 1/3 = 0.3333이다