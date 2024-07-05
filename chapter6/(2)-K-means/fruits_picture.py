# K-평균(k-means) 군집 알고리즘
# 비지도 학습에서 평균값을 자동으로 찾아준다

# 이 평균값이 클러스터의 중심에 위치하기 때문에
# 클러스터 중심(cluster center) 또는 센트로이드(centroid)라고 부른다

# k-평균 알고리즘의 작동 방식
# 1. 무작위로 k개의 클러스터 중심을 정한다
# 2. 각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정
# 3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경
# 4. 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복

# wget으로 데이터 다운로드
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

# 넘파이 np.load() 함수를 사용해 npy 파일을 읽어 넘파이 배열을 준비
# k-평균모델을 훈련하기 위해 (샘플 개수, 너비, 높이) 크기의 3차원 배열을 (샘플 개수, 너비 X 높이)크기를 가진 2차원 배열로 변경
import numpy as np

fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100 * 100)

# 사이킷런의 k-평균 알고리즘은 sklean.cluster 모듈 아래 KMeans 클래스에 구현되어 있다
# 이 클래스에서 설정할 매개변수는 클러스터 개수를 지정하는 n_clusters이다 
# 여기서는 클러스터 개수를 3으로 지정한다

# 사용법도 다른 클래스와 비슷하다
# 다만 비지도 학습이므로 fit() 메서드에서 타깃 데이터를 사용하지 않는다
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_2d)

# 군집된 결과는 KMeans 클래스 객체의 labels_ 속성에 저장된다
# labels_ 배열의 길이는 샘플 개수와 같다 
# 이 배열은 각 샘플이 어떤 레이블에 해당되는지 나타낸다

# n_cluster=3으로 지정했기 때문에 labels_ 배열의 값은 0, 1, 2 중 하나이다
print(km.labels_)
# [2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 0 2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 0 0 2 2 2 2 2 2 2 2 0 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1]

# 레이블값 0, 1, 2와 레이블 순서에는 어떤 의미도 없다
# 실제 레이블 0, 1, 2가 어떤 과일 사진을 주로 모았는지 알아보려면 직접 이미지를 출력하는 것이 최선이다

# 레이블 0, 1, 2로 모은 샘플의 개수 확인
print(np.unique(km.labels_, return_counts=True))
# (array([0, 1, 2], dtype=int32), array([111,  98,  91]))

# 첫 번째 클러스터(레이블 0)가 111개의 샘플 수집
# 두 번째 클러스터(레이블 1)가 98개의 샘플 수집
# 세 번째 클러스터(레이블 2)가 91개의 샘플 수집

# 각 클러스터가 어떤 이미지를 나타냈는지 그림으로 출력하기 위해 간단한 유틸리티 함수 draw_fruits() 만들기
