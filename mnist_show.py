import numpy as np
from mnist import load_mnist	# 모든 *.py 파일들을 하나의 폴더 내에 저장한다고 가정함.
from PIL import Image		# PIL: Python Image Library

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=False, flatten=False)

img = x_train[0]		# x_train.shape: (60000, 784)
label = t_train[0]		# t_train.shape: (60000,)
print(label)			# 5

print(img.shape)		# (28, 28)

img_show(img)