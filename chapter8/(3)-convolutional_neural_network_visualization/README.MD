가중치 시각화
합성곱 층의 가중치를 이미지로 출력하는것
합성곱 신경망은 주로 이미지를 다루기 떄문에 가중치가 시각적인 패턴을
학습하는지 알아볼 수 있다

특성 맵 시각화
합성곱 층의 활성화 출력을 이미지로 그리는 것
가중치 시각화와 함께 비교하여 각 필터가 이미지의 어느 부분을 활성화
시키는지 확인 가능

함수형 API
케라스에서 신경망 모델을 만드는 방법 중 하나
Model 클래스에 모델의 입력과 출력을 지정
전형적으로 입력은 Input() 함수를 사용하여 정의하고 출력은 
마지막 층의 출력으로 정의

1. TensorFlow
Model
케라스 모델을 만드는 클래스
첫 번째 매개변수인 inputs에 모델의 입력 또는 입력의 리스트를 지정
두 번째 매개변수인 outputs에 모델의 출력 또는 출력의 리스트를 지정
name 매개변수에 모델의 이름을 지정