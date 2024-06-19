# 판다스로 와인 데이터 불러오기
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine_csv_data')

# 와인 샘플 5개 출력
wine.head()
# 	alcohol	sugar	pH	class
# 0	9.4	1.9	3.51	0.0
# 1	9.8	2.6	3.20	0.0
# 2	9.8	2.3	3.26	0.0
# 3	9.8	1.9	3.16	0.0
# 4	9.4	1.9	3.51	0.0

# 레드와인, 화이트와인 구분은 이진 분류 문제이고 화이트 와인이 양성 클래스
# 즉 전체 와인 데이터에서 화이트 와인을 골라내는 문제

# 판다스의 info() 메서드
# 데이터프레임의 각 열의 데이터 타입과 누락된 데이터가 있는지 확인하는데 유용
wine.info()

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 6497 entries, 0 to 6496
# Data columns (total 4 columns):
#  #   Column   Non-Null Count  Dtype  
# ---  ------   --------------  -----  
#  0   alcohol  6497 non-null   float64
#  1   sugar    6497 non-null   float64
#  2   pH       6497 non-null   float64
#  3   class    6497 non-null   float64
# dtypes: float64(4)
# memory usage: 203.2 KB

# 출력 결과를 보면 총 6497개의 샘플이 있고 4개의 열은 모두 실숫값 
# Non-Null Count가 모두 6497이므로 누락된 값은 없다

# 누락된 값이 있다면 그 데이터를 버리거나 평균값으로 채운 후 사용할 수 있다
# 어떤 방식이 최선인지는 미리 알기 어렵다
# 두 가지 모두 시도해보아야 한다
# 여기에서도 항상 훈련 세트의 통계 값으로 테스트 세트를 변환해야 한다
# 즉 훈련 세트의 평균값으로 테스트 세트의 누락된 값을 채워야 한다

# 판다스의 describe()
# 열에 대한 간략한 통계를 출력한다 (최소, 최대, 평균값 등)
wine.describe()

#         alcohol	sugar	pH	class
# count	6497.000000	6497.000000	6497.000000	6497.000000
# mean	10.491801	5.443235	3.218501	0.753886      평균
# std	    1.192712	4.757804	0.160787	0.430779  표준편차
# min	    8.000000	0.600000	2.720000	0.000000  최소
# 25%	    9.500000	1.800000	3.110000	1.000000  1사분위수
# 50%	    10.300000	3.000000	3.210000	1.000000  중간값 / 2사분위수
# 75%	    11.300000	8.100000	3.320000	1.000000  3사분위수
# max	    14.900000	65.800000	4.010000	1.000000  최대

# 사분위수는 데이터를 순서대로 4등분 한 값이다 
# 예를 들어 2사분위수(중간값)는 데이터를 일렬로 늘어놓았을 때 정중앙의 값이다
# 만약 데이터 개수가 짝수개라 중앙값을 선택할 수 없다면 가운데 2개의 값의 평균을 사용한다

# 여기서 알 수 있는 것은 알코올 도수와 당도, PH 값의 스케일이 다르다는 것이다
# 사이킷런의 StandardScaler 클래스를 사용해 특성을 표준화하는 전처리 작업이 필요하다

# 판다스 데이터프레임을 넘파이 배열로 바꾸고 훈련 세트와 테스트 세트로 나눈다

# 타깃 데이터와 샘플 데이터 분리
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

# 훈련 세트와 테스트 세트 나누기
from sklearn.model_selection import train_test_split

# test_size를 지정하지 않으면 기본값으로 25%를 테스트 세트로 지정 
# 샘플이 충분히 많으므로 0.2로 지정하여 20%정도만 테스트 세트로 나눈다
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

# 훈련 세트와 테스트 세트의 크기 확인
print(train_input.shape, test_input.shape)
# (5197, 3) (1300, 3)

# StandardScaler 클래스를 사용해 훈련 세트를 전처리
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 로지스틱 회귀 모델 훈련
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(train_scaled, train_target)

print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
# 0.7808350971714451
# 0.7776923076923077

# 모델이 다소 과소적합되었다
# 이 문제를 해결하기 위해 규제 매개변수 C의 값을 바꿀수도 있고 solver 매개변수에서 다른 알고리즘을 선택할 수도 있다
# 또는 다항 특성을 만들어 추가할 수도 있다

