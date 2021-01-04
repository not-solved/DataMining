#       퍼셉트론 구현
def AND_p(x1, x2) :
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta :
        return 0
    elif tmp > theta :
        return 1

print(AND_p(0, 0))
print(AND_p(0, 1))
print(AND_p(1, 0))
print(AND_p(1, 1))

#       가중치와 편향 도입
import numpy as np
x = np.array( [0, 1] )
w = np.array( [0.5, 0.5] )      # 가중치 (weight)
b = -0.7                        # 편향   (bias)
print(w * x)
print(np.sum(w*x) + b)

def AND_w(x1, x2) :
   x = np.array([x1, x2])
   w = np.array([0.5, 0.5])
   b = -0.7
   tmp = np.sum(w*x) + b
   if tmp <= 0 :
       return 0
   else :
       return 1

def NAND(x1, x2) :
   x = np.array([x1, x2])
   w = np.array([-0.5, -0.5])
   b = 0.7
   tmp = np.sum(w*x) + b
   if tmp <= 0 :
       return 0
   else :
       return 1

def OR(x1, x2) :
   x = np.array([x1, x2])
   w = np.array([0.5, 0.5])
   b = -0.2
   tmp = np.sum(w*x) + b
   if tmp <= 0 :
       return 0
   else :
       return 1

def XOR(x1, x2) :
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND_w(s1, s2)
    return y

print(AND_w(1, 1))
print(NAND(1, 1))
print(OR(1, 1))
print(XOR(1, 0))