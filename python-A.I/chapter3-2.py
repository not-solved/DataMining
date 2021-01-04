#       MNIST 데이터 셋 (손글씨 숫자 인식)
import sys, os
import numpy as np
sys.path.append(os.pardir)          # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from mnist import load_mnist
from PIL import Image

def img_show(img) :
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten = True, normalize = False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

###################################################################################################

#       신경망의 추론 처리
import pickle
def sigmoid(x) :
    return 1 / (1 + np.exp(-x))
def softmax(x) : 
    C = np.max(x)
    exp_x = np.exp(x - C)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y


def get_data() :
    (x_train, t_train) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network() :
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x) : 
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)) :
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i] :
        accuracy_cnt += 1

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))

###################################################################################################

# 배치 처리
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
print(x.shape)
print(x[0].shape)
print(W1.shape)
print(W2.shape)
print(W3.shape)


x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size) :
    x_batch = x[i:i+batch_size]                         # x에서 100단위로 배치
    y_batch = predict(network, x_batch)                 # 회귀
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])      # t의 100단위와 y에서 뽑은 최대값과 비교

print("Accuracy : " + str(float(accuracy_cnt)/len(x)))
