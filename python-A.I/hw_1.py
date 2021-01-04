# 12151411_심경수

import numpy as np

def sigmoid(x) :                    # sigmoid function
    return 1 / (1 + np.exp(-x))

def init_network() :                # initialize neural network matrix
    network = {}
    network['W1'] = np.array([ [0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6] ])
    network['W2'] = np.array([ [0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9] ])
    return network

def forwarding(network, x) :        # neural network processing (forwarding, no back propogation)
    W1, W2 = network['W1'], network['W2']

    a1 = np.dot(W1, x)
    z1 = sigmoid(a1)
    a2 = np.dot(W2, z1)
    z2 = sigmoid(a2)
    return z2

input = np.array([0.9, 0.1, 0.8])
output = np.array([0.726, 0.708, 0.778])    # Target ( = ground truth)
network = init_network()

# print("input\t:", input)
# print("output\t:", forwarding(network, input))
print(forwarding(network, input))           # Actual