# 비지도 학습(unsupervised learning)
# 타깃이 없을 때 사용하는 머신러닝 알고리즘
# 데이터에 있는 무언가를 학습한다

# 과일 사진 데이터 다운로드
# 넘파이 배열의 기본 저장 포맷인 npy 파일로 저장되어 있다
# !는 코랩에서 셸 명령어 실행
!wget https://bit.ly/fruits_300_data -O fruits_300.npy

# 넘파이와 맷플롯립 임포트
import numpy as np
import matplotlib.pyplot as plt

# 넘파이로 npy 파일 로드
fruits = np.load('fruits_300.npy')

# 배열 크기 확인
print(fruits.shape)
# (300, 100, 100)

# 첫 번째 차원 = 샘플의 개수
# 두 번째 차원 = 이미지 높이
# 세 번째 차원 = 이미지 너비
# 각 픽셀은 넘파이 배열의 원소 하나에 대응
# 즉 이미지 크기는 100 X 100 이다

# 첫 번째 이미지의 첫 행 픽셀 출력
print(fruits[0, 0, :])
# [  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   2   1
#    2   2   2   2   2   2   1   1   1   1   1   1   1   1   2   3   2   1
#    2   1   1   1   1   2   1   3   2   1   3   1   4   1   2   5   5   5
#   19 148 192 117  28   1   1   2   1   4   1   1   3   1   1   1   1   1
#    2   2   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
#    1   1   1   1   1   1   1   1   1   1]

# 0에 가까울수록 검게 나타나고 높은 값은 밝게 표시된다

# 첫 번째 이미지를 그림으로 그리기
# 맷플롯립의 imshow() 함수를 사용하면 넘파이 배열로 저장된 이미지를 쉽게 그릴 수 있다
# 흑백 이미지이므로 cmap 매개변수를 gray로 지정
plt.imshow(fruits[0], cmap='gray')
plt.show()

# 흑백 사진은 보통 바탕이 밝고 물체가 짙은 색이다
# 불러온 사진의 흑백 이미지는 사진으로 찍은 이미지를 넘파이 배열로 변환할 때 반전시킨 것이다
# 사진의 흰 바탕(높은 값)은 검은색(낮은 값)으로 만들고 실제 사과가 있어 짙은 부분(낮은 값)은 밝은색(높은 값)으로 바꾸었다

# 이렇게 바꾼 이유
# 흰색 바탕일수록 값이 높다
# 컴퓨터는 높은 값에 집중할 것이므로 바탕에 집중할 것이다
# 알고리즘이 어떤 출력을 만들기 위해 곱셈, 덧셈을 한다
# 픽셀값이 0이면 출력도 0이 되어 의미가 없다 
# 픽셀값이 높으면 출력값도 커지기 때문에 의미를 부여하기 좋다
# 그렇기 때문에 컴퓨터는 바탕에 집중하게 된다

# 우리가 보는것과 컴퓨터가 처리하는 방식이 다르기 떄문에 종종 흑백 이미지를 이렇게 반전하여 사용한다

# 색을 반전 시켜 사과 이미지 그리기
plt.imshow(fruits[0], cmap='gray_r')
plt.show()

# 맷플롯립의 subplots()함수를 사용하면 여러 개의 그래프를 배열처럼 쌓을 수 있도록 도와준다
# subplots() 함수의 두 매개변수는 그래프를 쌓을 행과 열을 지정

# 반환된 axs는 2개의 서브 그래프를 담고 있는 배열

# 바나나와 파인애플 이미지 출력
fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()

# 픽셀값 분석을 위해
# 과일 하나의 2차원 데이터를 펼쳐 1차원 데이터로 만들기
      
# fruits[0:100] = 슬라이싱 연산자로 100개의 과일 선택
# reshape(-1, 100 * 100) = reshape() 메서드를 사용해 두 번째 차원과 세 번째 차원을 10000으로 합친다 
# (100*100 = 10000, 배열을 순차적으로 0부터 10000까지 순회하면서 일차원 배열 생성)
# 2차원 배열은 reshape(-1, 100, 100) 
# 첫 번째 차원을 -1로 지정하면 자동으로 남은 차원을 할당한다 (열을 알아서 재배열)
# 여기에서는 첫 번째 차원이 샘플 개수다
apple = fruits[0:100].reshape(-1, 100 * 100)
pineapple = fruits[100:200].reshape(-1, 100 * 100)
banana = fruits[200:300].reshape(-1, 100 * 100)

# 배열 크기 확인
print(apple.shape)
# (100, 10000)

# 샘플 평균값 계산
# 넘파이의 mean() 메서드 사용 
# 샘플마다 픽셀의 평균값을 계산해야 하므로 mean()메서드가 평균을 계산할 축을 지정해야한다
# axis=0으로 하면 첫 번째 축인 행을 따라 계산
# axis=1로 지정하면 두 번째 축인 열을 따라 계산

# 사과 배열의 평균값 계산
print(apple.mean(axis=1))
# [ 88.3346  97.9249  87.3709  98.3703  92.8705  82.6439  94.4244  95.5999
#   90.681   81.6226  87.0578  95.0745  93.8416  87.017   97.5078  87.2019
#   88.9827 100.9158  92.7823 100.9184 104.9854  88.674   99.5643  97.2495
#   94.1179  92.1935  95.1671  93.3322 102.8967  94.6695  90.5285  89.0744
#   97.7641  97.2938 100.7564  90.5236 100.2542  85.8452  96.4615  97.1492
#   90.711  102.3193  87.1629  89.8751  86.7327  86.3991  95.2865  89.1709
#   96.8163  91.6604  96.1065  99.6829  94.9718  87.4812  89.2596  89.5268
#   93.799   97.3983  87.151   97.825  103.22    94.4239  83.6657  83.5159
#  102.8453  87.0379  91.2742 100.4848  93.8388  90.8568  97.4616  97.5022
#   82.446   87.1789  96.9206  90.3135  90.565   97.6538  98.0919  93.6252
#   87.3867  84.7073  89.1135  86.7646  88.7301  86.643   96.7323  97.2604
#   81.9424  87.1687  97.2066  83.4712  95.9781  91.8096  98.4086 100.7823
#  101.556  100.7027  91.6098  88.8976]

# 히스토그램
# 값이 발생한 빈도를 그래프로 표시한것
# 보통 x축이 값의 구간(계급)이고 y축은 발생 빈도(도수)이다
# 엑셀이나 스프레드시트 등에서 막대그래프와 유사하다

# 히스토그램을 그려 평균값 분포도 파악
# 맷플롯립의 hist() 함수를 사용
# alpha 매개변수로 투명도 조절
# legend() 함수로 범례 생성
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineapple', 'banana'])
plt.show()

# 히스토그램 출력 결과를 보면 바나나 사진의 평균값은 40 아래에 집중되어 있다
# 사과와 파인애플은 90~100 사이에 많이 모여있다

# 바나나는 사과나 파인애플과 확실히 구분된다
# 바나나는 차지하는 영역이 작기 때문에 평균값이 작다

# 사과와 파인애플은 픽셀값만으로는 구분하기 쉽지 않다
# 형태가 대체로 동그랗고 사진에서 차지하는 크기도 비슷하기 떄문이다

# 샘플의 평균값이 아니라 픽셀별 평균값을 비교
# 전체 샘플에 대해 각 픽셀의 평균을 계산
# 세 과일은 모양이 다르므로 픽셀값이 높은 위치가 조금 다를 것이다

# 픽셀의 평균 계산 
# axis=0으로 지정
# 맷플롯립의 bar() 함수를 사용해 픽셀 10000개에 대한 평규낪을 막대그래프로 출력
# subplots() 함수로 3개의 서브 그래프를 만든다
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()

# 출력된 3개의 그래프를 보면 과일마다 값이 높은 구간이 다르다
# 사과는 사진 아래쪽으로 갈수록 값이 높아지고 
# 파인애플 그래프는 비교적 고르면서 높다
# 바나나는 확실히 중앙의 픽셀값이 높다

# 픽셀 평균값을 100 X 100 크기로 바꿔서 이미지처럼 출력하여 위 그래프와 비교하면 더 좋다
# 픽셀을 평균 낸 이미지를 모든 사진을 합쳐 놓은 대표 이미지로 생각할 수 있다

# 사과의 픽셀 평균값을 구하여 2차원 배열로 변환
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis=0).reshape(100, 100)
banana_mean = np.mean(banana, axis=0).reshape(100, 100)

fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')

plt.show()

# 평균값과 가까운 사진 고르기
# 절댓값 오차를 사용
# fruits 배열에 있는 모든 샘플에서 apple_mean을 뺀 절댓값의 평균을 계산

# 넘파이의 abs() 함수로 절댓값을 계산
# np.abs(-1)은 1을 반환
# 배열을 입력하면 모든 원소의 절댓값을 계산하여 입력과 동일한 크기의 배열을 반환
# 이 함수는 np.absolute() 함수의 다른 이름이다

# 다음 코드의 abs_diff는 (300, 100, 100) 크기의 배열이다
# 따라서 각 샘플에 대한 평균을 구하기 위해 axis에 두 번째, 세 번째 차원을 모두 지정 (사과, 파인애플, 바나나의 1개 과일 픽셀 평균값)
# 이렇게 계산한 abs_mean은 각 샘플의 오차 평균이므로 크기가 (300,)인 1차원 배열이다
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)

# apple_mean과 오차가 가장 적은 샘플 100개 고르기
# np.argsort() 함수는 작은 것에서 큰 순서대로 나열한 abs_mean배열의 인덱스를 반환
# 이 인덱스 중에서 처음 100개를 선택해 10 X 10 격자로 이루어진 그래프 출력

# 깔끔하게 이미지만 그리기 위해 axis('off')로 사용하여 좌표축을 그리지 않음
apple_index = np.argsort(abs_mean)[:100]

fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10+j]], cmap='gray_r')
        axs[i, j].axis('off')

plt.show()
# 100개 모두 사과 이미지만 출력된다

# 군집 (clustering)
# 비슷한 샘플끼리 그룹으로 모으는 작업
# 군집은 대표적인 비지도 학습 작업 중 하나

# 클러스터 (cluster)
# 군집 알고리즘에서 만든 그룹

# 실제 비지도 학습에서는 타깃값을 모르기 때문에 이처럼 샘플의 평균값을 미리 구할 수 없다
