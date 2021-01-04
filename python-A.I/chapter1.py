
#           numpy
import numpy as np

a = np.array( [1.0, 2.0, 3.0] )     # 기본적인 배열 포맷
print(a)
type(a)

b = np.array( [2.0, 4.0, 6.0] )
print(a + b)
print(a - b)        
print(a * b)        
print(a / b)
print(a ** b)               # 각 원소별로 산술연산 수행

c = np.array([ [1, 2], [3, 4] ])    # 2차원 배열
print(c)
print(c.shape)
print(c.dtype)

d = np.array([ [3, 0], [0, 6] ])    # 2차원 배열끼리 연산도 간단하게 가능
print(c + d)
print(c * d)

D = d.flatten()
print(D)                    # 2차원 배열 d를 한 줄로 표현
print(D[np.array([0, 2])]) # 0, 2번째 인덱스의 데이터만 출력


#           matplotlib  데이터 시각화 모듈 ( 그래프 등...)
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)    # (0에서 6까지 0.1 간격으로 생성)
y1 = np.sin(x)              # 함수 1
y2 = np.cos(x)              # 함수 2

plt.plot(x, y1, label="sin")                    # x축 데이터, y축 데이터, 라벨 이름
plt.plot(x, y2, linestyle="--", label="cos")    # x축 데이터, y축 데이터, 선 스타일, 라벨 이름

plt.xlabel("X")                                 # x축 이름
plt.ylabel("Y")                                 # y축 이름
plt.title('sin & cos')                          # 그래프 타이틀
plt.legend()
plt.show()                                      # 출력

#              matplotlib   이미지 표시하기
from matplotlib.image import imread

img = imread('lena.png')    # 먼저 경로를 설정한 후 열어야 한다! (여기서는 실행 안됨)

plt.imshow(img)             # 이미지 표시 준비
plt.show()                  # 출력