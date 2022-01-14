from functions import sigmoid, softmax
import numpy as np

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # (input_size, hidden_size, output_size) = (784, 50, 10)

        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1']  = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2']  = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        # x.shape: (600, 784) 또는 (100, 784)

        a1 = np.dot(x, W1) + b1		# W1.shape: (784, 50)
        z1 = sigmoid(a1)			# z1.shape:  (600, 50) 또는 (100, 50)
        a2 = np.dot(z1, W2) + b2		# W2.shape: (50, 10)
        y = softmax(a2)			# y.shape:    (600, 10) 또는 (100, 10)
        
        return y

    def accuracy(self, x, t):
        # x.shape: (600, 784) 또는 (100, 784)
        # t.shape: (600, 10) 또는 (100, 10)

        y = self.predict(x)			# y.shape: (600, 10) 또는 (100, 10)
        y = np.argmax(y, axis=1)		# y.shape: (600, ) 또는 (100, )
        t = np.argmax(t, axis=1)		# t.shape: (600, ) 또는 (100, )
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        ##########    순전파 파트    ###########
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y  = softmax(a2)

        ##########    역전파 파트    ###########
        dLda2 = (y - t)
        grads['W2'] = np.dot(z1.T, dLda2)
        grads['b2'] = dLda2[0]
        
        dLdz1 = np.dot(dLda2, W2.T)
        dLda1 = z1 * (1-z1) * dLdz1

        grads['W1'] = np.dot(x.T, dLda1)
        grads['b1'] = dLda1[0]

        return grads
