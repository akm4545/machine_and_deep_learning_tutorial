합성곱
밀집층과 비슷하게 입력과 가중치를 곱하고 절편을 더하는 선형 계산
하지만 밀집층과 달리 각 합성곱은 입력 전체가 아니라 일부만 사용하여 선형 계산을 수행

필터
합성곱 층의 필터는 밀집층의 뉴런에 해당
필터의 가중치와 절편을 종종 커널이라고 부른다
자주 사용되는 커널의 크기는 (3, 3) 또는 (5, 5)이다
커널의 깊이는 입력의 깊이와 같다

특성 맵
합성곱 층이나 풀링 층의 출력 배열을 의미
필터 하나가 하나의 특성 맵을 만든다
합성곱 층에서 5개의 필터를 적용하면 5개의 특성 맵이 만들어진다

패딩
합성곱 층의 입력 주위에 추가한 0으로 채워진 픽셀이다
패딩을 사용하지 않는 것을 밸리드 패딩이라고 한다
합성곱 층의 출력 크기를 입력과 동일하게 만들기 위해 입력에 패딩을 추가하는 것을 세임 패딩이라고 한다

스트라이드
합성곱 층에서 필터가 입력 위를 이동하는 크기
일반적으로 스트라이드는 1픽셀을 사용

풀링 
가중치가 없고 특성 맵의 가로세로 크기를 줄이는 역할
대표적으로 최대 풀링과 평균 풀링이 있으며 (2, 2) 풀링으로 입력을 절반으로 줄인다