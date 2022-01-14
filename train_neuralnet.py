import numpy as np
import matplotlib.pyplot as plt		# 그래프 표시를 위한 library
from mnist import load_mnist
from three_layer_net import ThreeLayerNet	# (784, 50, 10) 형태의 2층 신경망

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True, flatten=True)

print("x_train.shape: ", x_train.shape)	# (600, 784)
print("t_train.shape: ", t_train.shape)		# (600, 10)
print("x_test.shape: ", x_test.shape)		# (100, 784)
print("t_test.shape: ", t_test.shape)		# (100, 10)

# 2층 신경망 초기화
network = ThreeLayerNet(input_size=784, hidden_size=50, output_size=10)

train_size = x_train.shape[0]		# 600 (= 학습 데이터의 수)
num_of_epochs = 20			# 전체 학습 데이터를 10회 반복 사용
learning_rate = 0.3			# 학습률 설정

train_acc_list = [0.1]			# 학습 데이터에 대한 정확도 저장을 위한 list
test_acc_list = [0.1]			# 시험 데이터에 대한 정확도 저장을 위한 list

for epoch_num in range(1, num_of_epochs+1):		# 에폭 수만큼 전체 데이터 반복 적용
    for data_num in range(train_size):			# 편의상, 데이터를 1개씩 처리

        # 기울기 (가중치 매개변수에 대한 손실 함수의 편미분) 계산
        grad = network.gradient(x_train[[data_num]], t_train[[data_num]])

        # x_train[[data_num]].shape  (1, 784)
        # x_train[data_num].shape  (784, )
    
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]	# 가중치 매개변수 업데이트

    train_acc = network.accuracy(x_train, t_train)		# 학습 데이터에 대한 정확도 계산
    test_acc = network.accuracy(x_test, t_test)		# 시험 데이터에 대한 정확도 계산

    train_acc_list.append(train_acc)			# 학습 데이터에 대한 정확도 리스트 갱신
    test_acc_list.append(test_acc)			# 시험 데이터에 대한 정확도 리스트 갱신

    # 정확도 결과를 text 형태로 출력
    print("epoch_num=", epoch_num, ", data_num=", data_num, \
          ", train_acc=", train_acc, ", test_acc=", test_acc)

# 정확도 결과를 graph 형태로 출력
x = np.arange(len(train_acc_list))	# length=(num_of_epochs)+1 (dummy 데이터 1개 포함)

plt.plot(x, train_acc_list, label='train acc')		# 학습 데이터 정확도 출력
plt.plot(x, test_acc_list, label='test acc', linestyle='--')	# 시험 데이터 정확도 출력

plt.xlabel("epochs")		# x축 제목 표시
plt.ylabel("accuracy")		# y축 제목 표시

plt.ylim(0, 1.0)		# y축 범위 지정
plt.legend(loc='lower right')	# 범례 표시 및 위치 지정
plt.show()
