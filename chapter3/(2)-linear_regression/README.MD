k-최근접 이웃 회귀
훈련 세트 범위 밖의 샘플을 예측할 수 없다 
k-최근접 이웃 회귀는 아무리 멀리 떨어져 있더라도 무조건 가장 가까운 샘플의 타깃을 평균하여 예측

선형 회귀
특성과 타깃 사이의 관계를 가장 잘 나타내는 선형 방정식을 찾는다
특성이 하나면 직선 방정식이 된다

선형 회귀가 찾은 특성과 타깃 사이의 관계는 선형 방정식의 계수 또는 가중치에 저장된다
머신러닝에서는 종종 가중치는 방정식의 기울기와 절편을 모두 의미하는 경우가 많다

가장 잘 맞는 직선의 방정식을 찾는다는 것은 최적의 기울기와 절편을 구한다는 의미다
이 값은 선형 회귀 모델의 coef_와 intercept_ 속성에 저장되어 있다

k-최근접 이웃 회귀와 다르게 훈련 세트를 벗어난 범위의 데이터도 잘 예측했지만 모델이 단순하여 이상한 수치가 나올 수 있다

모델 파라미터
선형 회귀가 찾은 가중치처럼 머신러닝 모델이 특성에서 학습한 파라미터

scikit-learn
LinearRegression
사이킷런의 선형 회귀 클래스
fit_intercept 매개변수를 False로 지정하면 절편을 학습하지 않는다 기본값은 True
학습된 모델의 coef_ 속성은 특성에 대한 계수를 포함한 배열이다 
즉 이 배열의 크기는 특성의 개수와 같다
intercept_ 속성에는 절편이 저장되어 있다