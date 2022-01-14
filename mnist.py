import urllib.request				# 다운로드 관련 라이브러리
import gzip					# 압축 관련 라이브러리
import pickle				# pickle 관련 라이브러리
import os					# OS 관련 라이브러리
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
down_files = {				# down_files dictionary
    'train_img':'train-images-idx3-ubyte.gz',		# 학습 데이터 (이미지)
    'train_label':'train-labels-idx1-ubyte.gz',		# 학습 데이터 (label)
    'test_img':'t10k-images-idx3-ubyte.gz',		# 테스트 데이터 (이미지)
    'test_label':'t10k-labels-idx1-ubyte.gz'		# 테스트 데이터 (label)
}
dataset_dir = os.path.dirname(os.path.abspath(__file__))	# D:/zDeep/zTest
pkl_file = dataset_dir + "/mnist.pkl"			# D:/zDeep/zTest/mnist.pkl

train_num = 60000				# 학습 데이터의 수
test_num = 10000				# 테스트 데이터의 수
img_size = 784				# 784 = 28*28


def _download(file_name):		# 각각의 파일을 다운로드
    full_f_name = dataset_dir + "/" + file_name	# 가령, D:/zDeep/zTest/train-images-idx3-ubyte.gz
    
    if os.path.exists(full_f_name):		# 이미 해당 파일 존재하면, 잔여 과정 생략.
        return					

    print("Downloading " + file_name + " ... ")		# 다운로드 시작 메시지
    urllib.request.urlretrieve(url_base+file_name, full_f_name)	# 실제 다운로드 과정
    print("Done")				# 다운로드 완료 메시지

def download_mnist():				# 4개의 MNIST 파일을 다운로드
    for v in down_files.values():			# 4개 파일에 대해 iteration
       _download(v)
        
def _load_label(file_name):		# 레이블 파일 내용을 numpy array 형태로 변환하여 저장
    full_f_name = dataset_dir + "/" + file_name	# 가령, D:/zDeep/zTest/train-labels-idx1-ubyte.gz
    
    print("Converting " + file_name + " to NumPy Array ...")	# 변환 시작 메시지
    with gzip.open(full_f_name, 'rb') as f:		# 파일 open (및 자동 close)
            labels = np.frombuffer(f.read(), np.uint8, offset=8)	# un-signed 8-bit 정수 형태로 저장
    print("Done")				# 변환 완료 메시지
    
    return labels

def _load_img(file_name):		# 이미지 파일 내용을 numpy array 형태로 변환하여 저장

    full_f_name = dataset_dir + "/" + file_name	# 가령, D:/zDeep/zTest/train-images-idx3-ubyte.gz
    
    print("Converting " + file_name + " to NumPy Array ...")	# 변환 시작 메시지
    with gzip.open(full_f_name, 'rb') as f:		# 파일 open (및 자동 close)
            data = np.frombuffer(f.read(), np.uint8, offset=16)	# un-signed 8-bit 정수 형태로 저장
    data = data.reshape(-1, img_size)			# (60000, 784) 또는 (10000, 784)
    print("Done")				# 변환 완료 메시지
    
    return data
    
def _convert_numpy():			# MNIST 파일 내용을 numpy array 형태로 변환하여 저장

    dataset = {}					# empty dictionary
    dataset['train_img'] =  _load_img(down_files['train_img'])		# 학습 데이터 (이미지)
    dataset['train_label'] = _load_label(down_files['train_label'])	# 학습 데이터 (label)
    dataset['test_img'] = _load_img(down_files['test_img'])		# 테스트 데이터 (이미지)
    dataset['test_label'] = _load_label(down_files['test_label'])		# 테스트 데이터 (label)
    
    return dataset


def init_mnist():			# MNIST 파일들을 다운로드하여 피클 파일로 저장
    download_mnist()			# 웹 사이트로부터 MNIST 파일들 다운로드
    dataset = _convert_numpy()		# MNIST 데이터를 dictionary 형태로 변환하여 저장
    print("Creating pickle file ...")
    with open(pkl_file, 'wb') as f:		# D:/zDeep/zTest/mnist.pkl
        pickle.dump(dataset, f, -1)		# 학습/시험 데이터를 피클 파일로 저장
    print("Done!")			

def _change_one_hot_label(X):		# 가령, label이 5일 경우, [0 0 0 0 0 1 0 0 0 0]로 변환
    T = np.zeros((X.size, 10), int)		# 가령, X:(60000, )이면, T:(60000, 10)

    for idx in range(T.shape[0]):
        T[idx][X[idx]] = 1			# 가령, idx=0일 때, X[idx]=5
        
    return T

def load_mnist(normalize=False, one_hot_label=False, flatten=True):
    if not os.path.exists(pkl_file):			# D:/zDeep/zTest/mnist.pkl
        init_mnist()
        
    with open(pkl_file, 'rb') as f:			# D:/zDeep/zTest/mnist.pkl
        dataset = pickle.load(f)			# 저장된 피클 파일을 읽어 옴.
    
    if normalize:		# [0, 255] 구간을 [0, 1] 구간으로 정규화
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)	# uint8  float32
            dataset[key] /= 255.0			# (60000, 784) 각 요소에 broadcast
            
    if one_hot_label:		# 가령, label이 5일 경우, [0 0 0 0 0 1 0 0 0 0] 형태로 변환
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    
    if not flatten:		# 가령, (60000*784) 형태를 (60000, 28, 28) 형태로 변환
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 

if __name__ == '__main__':
    init_mnist()	# MNIST 데이터를 다운로드 받고, 피클 파일로 저장하는 과정까지 진행.