import numpy as np

#       계단 함수
def step_function(x) :
    y = x > 0
    return y.astype(np.int)     # 입력 x가 배열일 경우 y도 배열로 하며 각 값의 결과를 정수 형태로 반환

x = np.array([-1.0, 1.0, 2.0])
y = x > 0
print(x)
print(y)
print(step_function(x))

#       활성화 함수 (sigmoid function)
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

#       ReUL 함수 (Rectified Linear Unit)
def ReUL(x) :
    return np.maximum(0, x)

#       다차원 배열
A = np.array([ [1, 2], [3, 4], [5, 6] ])
print(A)
print(np.ndim(A))       # A의 차원 수 확인
print(A.shape)          # A의 배열 형태 ( n*m ) 확인
print(A.shape[0])       # A의 0번째 열의 형태 반환

#################################################################################################################

#       행렬의 내적 (행렬 곱)
A = np.array([ [1,2], [3,4] ])
print(A.shape)
B = np.array([ [5,6], [7,8] ])
print(B.shape)
print(np.dot(A, B))     # dot product (내적 계산)

#       신경망의 내적
X = np.array([1, 2])
print(X.shape)
W = np.array([ [1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)
Y = np.dot(X, W)
print(Y)

#################################################################################################################

#       3층 신경망 구현하기
X = np.array([ [1.0, 0.5]])
W1 = np.array([ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1)
print(Z1)               # 신경망 전달 1단계 (input layer)

W2 = np.array([ [0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(A2)
print(Z2)               # 신경망 전달 2단계

def identity_function(x) :
    return x
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) # (Y = A3)
print(Y)

#################################################################################################################

#       구현 정리
def init_network() :
    network = {}
    network['W1'] = np.array([ [0.1, 0.3, 0.5], [0.2, 0.4, 0.6] ])
    network['W2'] = np.array([ [0.1, 0.4], [0.2, 0.5], [0.3, 0.6] ])
    network['W3'] = np.array([ [0.1, 0.3], [0.2, 0.4]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['b2'] = np.array([0.1, 0.2])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(a1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
#######################################################################################################

#       출력층 설계
#       소프트맥스 함수
def softmax_function(a) :       # 소프트맥스 함수 (모든 입력신호를 받아 하나의 출력을 낸다)
                                # 주의 : 아주 큰 값은 표현 불가 => 큰 값끼리 나눗셈 : 수치가 불안정해짐
    C = np.max(a)               # 추가 : 최댓값 지정           => 출력을 "확률"로 해석 가능
    exp_a = np.exp(a - C)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y