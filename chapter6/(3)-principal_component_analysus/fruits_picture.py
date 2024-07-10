# 차원(dimension)
# 데이터가 가진 속성을 특성이라 부른다
# 과일 사진의 경우 10000개의 픽셀이 있기 떄문에 10000개의 특성이 있는 셈이다
# 머신러닝에서는 이런 특성을 차원이라고도 부른다
# 10000개의 특성은 결국 10000개의 차원이다

# 차원 축소(dimension reduction)알고리즘
# 비지도 학습 작업 중 하나
# 차원 축소는 데이터를가장 잘 나타내는 일부 특성을 선택하여 데이터 크기를 줄이고
# 지도 학습 모델의 성능을 향상시킬 수 있는 방법

# 또한 줄어든 차원에서 다시 원본 차원으로 손실을 최대한 줄이면서 복원할 수도 있다

# 주성분 분석(principal component analysis)
# 대표적인 차원 축소 알고리즘 
# PCA라고도 부른다
# 주성분 분석은 데이터에 있는 분산이 큰 방향을 찾는 것으로 이해할 수 있다
# 분산은 데이터가 널리 퍼져있는 정도를 말한다
# 분산이 큰 방향이란 데이터를 잘 표현하는 어떤 벡터라고 생각할 수 있다

# 이 벡터를 주성분(principal component)라고 부른다
# 이 주성분 벡터는 원본 데이터에 있는 어떤 방향이다
# 따라서 주성분 벡터의 원소 개수는 원본 데이터셋에 있는 특성 개수와 같다
# 하지만 원본 데이터는 주성분을 사용해 차원을 줄일 수 있다

# 주성분은 원본 차원과 같고 주성분으로 바꾼 데이터는 차원이 줄어든다
# 주성분이 가장 분산이 큰 방향이기 때문에 주성분에 투영하여 바꾼 데이터는 원본이 가지고 있는
# 특성을 가장 잘 나타내고 있을 것이다

# 첫 번째 주성분을 찾은 다음 이 벡터에 수직이고 분산이 가장 큰 다음 방향을 찾는다
# 이 벡터가 두 번째 주성분이다 

# 일반적으로 주성분은 원본 특성의 개수만큼 찾을 수 있다
# 기술적인 이유로 주성분은 원본 특성의 개수와 샘플 개수 중 작은 값만큼 찾을 수 있다
# 일반적으로 비지도 학습은 대량의 데이터에서 수행하기 때문에 원본 특성의 개수만큼 찾을 수 있다고 말한다

# 과일 사진 데이터 다운로드 -> 넘파이 배열로 적재
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100 * 100)

# 사이킷런은 sklearn.decomposition 모듈 아래 PCA 클래스로 주성분 분석 알고리즘을 제공한다
# PCA 클래스의 객체를 만들 때 n_components 매개변수에 주성분의 개수를 지정해야 한다
# k-평균과 마찬가지로 비지도 학습이기 떄문에 fit() 메서드에 타깃값을 제공하지 않는다
from sklearn.decomposition import PCA

pca = PCA(n_components=50)
pca.fit(fruits_2d)

# PCA 클래스가 찾은 주성분은 components_속성에 저장되어 있다
# 속성 배열의 크기 확인
print(pca.components_.shape)
# (50, 10000)

# n_components=50으로 지정했기 때문에 pca.components_ 배열의 첫 번째 차원이 50이다
# 즉 50개의 주성분을 찾았다
# 두 번째 차원은 항상 원본 데이터의 특성 개수와 같은 10000이다

# 원본 데이터와 차원이 같으므로 주성분을 100 X 100 크기의 이미지처럼 출력해 볼 수 있다

# 주성분 이미지 출력
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
    n = len(arr) 
    rows = int(np.ceil(n / 10))    
    cols = n if rows < 2 else 10

    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)

    for i in range(rows):
        for j in range(cols):
            if i * 10 + j < n: 
                axs[i, j].imshow(arr[i * 10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    
    plt.show()

draw_fruits(pca.components_.reshape(-1, 100, 100))

# 이 주성분은 원본 데이터에서 가장 분산이 큰 방향을 순서대로 나타낸 것이다
# 한편으로는 데이터 셋에 있는 어떤 특징을 잡아낸 것처럼 생각할 수도 있다

# 주성분을 찾았으므로 원본 데이터를 주성분에 투영하여 특성의 개수를 10000개에서 50개로 줄일 수 있다
# 이는 마치 원본 데이터를 각 주성분으로 분해하는 것으로 생각할 수 있다

# PCA의 transform() 메서드를 사용해 원본 데이터의 차원을 50개로 줄이기
print(fruits_2d.shape)
# (300, 10000)
# fruits_2d는 (300, 10000) 크기의 배열이다
# 10000개의 픽셀(특성)을 가진 300개의 이미지이다

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
# (300, 50)
# 50개의 주성분을 찾은 PCA 모델을 사용해 이를 (300, 50) 크기의 배열로 변환했다
# 이제 fruits_pca 배열은 50개의 특성을 가진 데이;터다

# 데이터가 1/200로 줄어들었다

# 원본 데이터 재구성
# 10000개의 특성을 50개로 줄였기 때문에 어느 정도 손실이 발생할 수밖에 없다
# 하지만 최대한 분산이 큰 방향으로 데이터를 투영했기 때문에 원본 데이터를 상당 부분 재구성할 수 있다

# PCA 클래스는 이를 위해 inverse_transform() 메서드를 제공한다
# 50개의 차원으로 축소한 fruits_pca 데이터를 전달해 10000개의 특성을 복원
fruits_inverse = pca.inverse_transform(fruits_pca)
print(fruits_inverse.shape)
# (300, 10000)

# 복원된 특성을 100 X 100 크기로 바꾸어 100개씩 나누어 출력
fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)

for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")

# 설명된 분산(explained variance)
# 주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값
# PCA 클래스의 explained_variance_ratio_에 각 주성분의 설명된 분산 비율이 기록되어 있다
# 당연히 첫 번째 주성분의 설명된 분산이 가장 크다
# 이 분산 비율을 모두 더하면 50개의 주성분으로 표현하고 있는 총 분산 비율을 얻을 수 있다
print(np.sum(pca.explained_variance_ratio_))
# 0.9215050257986497

# 92%가 넘는 분산을 유지하고 있다
# 앞에서 50개의 특성에서 원본 데이터를 복원했을 때 원본 이미지의 품질이 높았던 이유이다
# 설명된 분산의 비율을 그래프로 그려 보면 적절한 주성분의 개수를 찾는 데 도움이 된다
plt.plot(pca.explained_variance_ratio_)
plt.show()

# 그래프를 보면 처음10개의 주성분이 대부분의 분산을 표현하고 있다
# 그 다음부터는 각 주성분이 설명하고 있는 분산은 비교적 작다

# 로지스틱 회귀 모델로 3개의 과일 사진을 분류
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

# 타깃값 생성
target = np.array([0] * 100 + [1] * 100 + [2] * 100)

# 원본 데이터를 사용하고 cross_validate()로 교차 검증 수행
from sklearn.model_selection import cross_validate

scores = cross_validate(lr, fruits_2d, target)

print(np.mean(scores['test_score']))
# 0.9966666666666667
print(np.mean(scores['fit_time']))
# 1.9659787654876708
# 각 교차 검증 폴드의 훈련 시간

# PCA로 축소한 fruits_pca 사용
scores = cross_validate(lr, fruits_pca, target)

print(np.mean(scores['test_score']))
# 1.0
print(np.mean(scores['fit_time']))
# 0.04012737274169922

# 50개의 특성만 사용했는데도 정확도가 100%이고 훈련 시간은 0.04초이다
# PCA로 훈련 데이터의 차원을 축소하면 저장 공간뿐만 아니라 머신러닝 모델의 훈련
# 속도도 높일 수 있다

# PCA 클래스를 사용할 때 n_components 매개변수에 주성분의 개수를 지정했다
# 이 대신 원하는 설명된 분산의 비율을 입력할 수도 있다
# PCA 클래스는 지정된 비율에 도달할 때까지 자동으로 주성분을 찾는다

# 설명된 분산의 50%에 달하는 주성분을 찾는 PCA 모델
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)

# 찾은 주성분 출력
print(pca.n_components_)
# 2

# 단 2개의 특성만으로 원본 데이터에 있는 분산의 50%를 표현할 수 있었다

# 해당 모델로 원본 데이터 변환
# 주성분이 2개이므로 변환된 데이터의 크기는 (300, 2)가 될 것이다
fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)
# (300, 2)

# 2개의 특성만 사용한 교차 검증 결과 출력
scores = cross_validate(lr, fruits_pca, target)

print(np.mean(scores['test_score']))
# 0.9933333333333334
print(np.mean(scores['fit_time']))
# 0.04055919647216797

# 코드를 입력하면 로지스틱 회귀 모델이 완전히 수렴하지 못했으니 반복 횟수를 증가하는 경고가 출력된다
# 하지만 교차 검증의 결과가 충분히 좋기 때문에 무시해도 좋다

# 차원 축소된 데이터를 사용해 k-평균 알고리즘으로 클러스터 찾기
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)

print(np.unique(km.labels_, return_counts=True))
# (array([0, 1, 2], dtype=int32), array([110,  99,  91]))

# 클러스터는 각각 110개, 99개 91개의 샘플을 포함하고 있다
# 이는 원본 데이터를 사용했을 때와 거의 비슷한 결과다

# KMeans가 찾은 레이블을 사용해 과일 이미지 출력
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")

# 훈련 데이터의 차원을 줄이면 또 하나 얻을 수 있는 장점은 시각화다
# 3개 이하로 차원을 줄이면 화면에 풀력하기 비교적 쉽다
# fruits_pca 데이터는 2개의 특성이 있기 때문에 2차원으로 표현할 수 있다
# 앞에서 찾은 km.lables_를 사용해 클러스터별로 나누어 산점도 출력
for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:,0], data[:,1])
plt.legend(['apple', 'banana', 'pineapple'])
plt.show()

# 산점도를 출력하면 각 클러스터의 산점도가 잘 구분된다
# 산점도를 보면 사과와 파인애플 클러스터의 경계가 가깝게 붙어 있다
# 이 두 클러스터의 샘플은 몇 개가 혼동을 일으키기 쉽다
# 데이터를 시각화하면 예상치 못한 통찰을 얻을 수 있다
# 그런 면에서 차원 축소는 매우 중요한 도구 중 하나다