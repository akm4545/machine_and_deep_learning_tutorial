1. TensorFlow
Conv2D
입력의 너비와 높이 방향의 합성곱 현산을 구현한 클래스
첫 번째 매개변수는 합서곱 필터의 개수
kernel_size 매개변수는 필터의 커널 크기를 지정 
가로세로 크기가 같은 경우 정수 하나로, 다른 경우 정수의 튜플로 지정 가능
일반적으로 커널의 가로세로 크기는 동일하다
커널의 깊이는 입력의 깊이와 동일하기 때문에 따로 지정하지 않는다
strides 매개변수는 필터의 이동 간격을 지정
가로세로 크기가 같은 경우 정수 하나로 다른 경우 정수의 튜플로 지정
일반적으로 가로세로 스트라이드 크기는 동일하다 기본값 1
padding 매개변수는 입력의 패딩 타입을 지정 
기본값 valid는 패딩을 하지 않는다
same은 합성곱 층의 출력의 가로세로 크기를 입력과 동일하게 맞추도록 입력에 패딩을 추가
activation 매개변수는 합성곱 층에 적용할 활성화 함수를 지정

MaxPooling2D
입력의 너비와 높이를 줄이는 풀링 연산을 구현한 클래스
첫 번째 매개변수는 풀링의 크기를 지정 
가로세로 크기가 같은 경우 정수 하나로, 다른 경우 정수의 튜플로 지정
일반적으로 풀링의 가로세로 크기는 같게 지정
strides 매개변수는 풀링의 이동 간격을 지정
기본값은 풀링의 크기와 동일 즉 입력 위를 겹쳐서 풀링하지 않는다
padding 매개변수는 입력의 패딩 타입을 지정 
기본값 valid는 패딩을 하지 않는다
same은 합성곱 층의 출력의 가로세로 크기를 입력과 동일하게 맞추도록 입력에 패딩을 추가

plot_model()
케라스 모델 구조를 주피터 노트북에 그리거나 파일로 저장
첫 번째 매개변수에 케라스 모델 객체를 전달
to_file 매개변수에 파일 이름을 지정하면 그림을 파일로 저장
show_shapes 매개변수를 True로 지정하면 층의 입력, 출력 크기를 표시
기본값 False
show_layer_names 매개변수를 True로 지정하면 층 이름을 출력 
기본값 True

2. matplotlib
bar()
막대그래프를 출력
첫 번째 매개변수에 x축의 값을 리스트나 넘파이 배열로 전달
두 번째 매개변수에 y축 값을 리스트나 넘파이 배열로 전달
width 매개변수에 막대의 두께를 지정 기본값 0.8