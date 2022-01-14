import numpy as np
import pickle
from mnist import load_mnist
from functions import sigmoid, softmax	# sigmoid, softmax 등 주요 함수 구현 라이브러리
from PIL import Image			# PIL: Python Image Library

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():			# 긴 라인은 back-slash 사용하여 표현할 수 있음.
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=False, flatten=True)
    return x_test, t_test			# 추론 파트만 테스트하므로 시험 데이터만 리턴.

def init_network():
    with open("sample_weight.pkl", 'rb') as f:	# 어느 정도 학습이 완료된 가중치 파라미터
        network = pickle.load(f)		# 학습된 가중치가 딕셔너리 형태로 저장되어 있음.
    return network

def predict(network, x):			# x: (784, )

    W1 = network['W1']			# (784, 50)
    W2 = network['W2']			# (50, 100)
    W3 = network['W3']			# (100, 10)

    b1 = network['b1']			# (50, )
    b2 = network['b2']			# (100, )
    b3 = network['b3']			# (10, )

    a1 = np.dot(x, W1) + b1		# (50, )
    z1 = sigmoid(a1)			# (50, )

    a2 = np.dot(z1, W2) + b2		# (100, )
    z2 = sigmoid(a2)			# (100, )

    a3 = np.dot(z2, W3) + b3		# (10, )
    y = softmax(a3)			# (10, )

    return y

x, t = get_data()		# brief 버전 사용 시... x.shape: (100, 784), t.shape: (100, )
print("len(x)=", len(x))		# brief 버전 사용 시... len(x)=100

network = init_network()	# 이미 학습이 어느 정도 진행된 가중치 파라미터 load
accuracy_cnt = 0

p = np.zeros_like(t, np.uint8)	# t와 같은 차원/형상의 zero 배열 생성

for i in range(len(x)):
    y = predict(network, x[i])	# 추론 (순전파) 과정 진행
    p[i] = np.argmax(y)		# 가장 높은 점수에 대응하는 인덱스 return

    if p[i] == t[i]:		# hit case
        accuracy_cnt += 1
    else:			# miss case
        print("i=", i, " p[i]=", p[i], " t[i]=", t[i])
        img = 255 * ( x[i].reshape(28, 28) )		# Denormalization
        if (i<100):
            img_show(img)

print("Accuracy:", float((accuracy_cnt) / len(x)) )	# 정확도: 96%
