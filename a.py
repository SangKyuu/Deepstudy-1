import os
from glob import glob
import numpy as np
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split


# from PIL import Image
# from matplotlib import pyplot as plt


## 질문1 : 구조가 덜 복잡하게 load 하는 법??

## 질문2 : shuffle data 어디서 shuffle???
## 폴더 별 data가 일정하게 섞이려면 shuffle을 load_data에서 해야하는지?
def load_data():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(root_dir, 'data', 'train')
    test_path = os.path.join(root_dir, 'data', 'test')

    WIDTH = 600
    HEIGHT = 600
    channel = 3
    train_len, test_len = 0, 0

    ## 질문3 : 파일 길이 구하려고 별도의 for문 써야하나??? / nparray 사용해야 하나.. (list로 하니 shape 안 맞다고 함)
    for n, label in enumerate(os.listdir(train_path)):  ## 0,1,2,3
        tr_path = os.path.join(train_path, label)  # 각 category별 폴더 경로 얻기
        te_path = os.path.join(test_path, label)

        train_file = glob(os.path.join(tr_path, "*.png"))
        test_file = glob(os.path.join(te_path, "*.png"))

        train_len += len(train_file)
        test_len += len(test_file)

    train = np.ndarray(shape=(train_len, WIDTH, HEIGHT, channel), dtype=np.float32)
    test = np.ndarray(shape=(test_len, WIDTH, HEIGHT, channel), dtype=np.float32)
    train_label = []
    test_label = []
    idx_train, idx_test = 0, 0

    # root에 있는 폴더 list 가져오기
    # 하부 폴더 : os.walk 참조
    for n, label in enumerate(os.listdir(train_path)):  ## 0,1,2,3
        tr_path = os.path.join(train_path, label)  # 각 category별 폴더 경로 얻기
        te_path = os.path.join(test_path, label)

        train_file = glob(os.path.join(tr_path, "*.png"))
        test_file = glob(os.path.join(te_path, "*.png"))

        for m, img in enumerate(train_file):  ## 사진 갯수만큼 반복
            # Read and resize image
            im = load_img(img, target_size=(WIDTH, HEIGHT))
            train[idx_train] = im
            idx_train += 1
            train_label.append(label)

        for m, img in enumerate(test_file):
            im = load_img(img, target_size=(WIDTH, HEIGHT))
            test[idx_test] = im
            idx_test += 1
            test_label.append(label)

    return train, test, train_label, test_label


def batch_data(train, test, train_label, test_label):
    # shuffle data 어디서 shuffle??? (질문2)
    idx = np.randint(0, len(train))

    for i in range(1, epoch + 1):
        # # Feed Batches
        for n, img in enumerate(train[:batch_size]):
            yield batch_image[n, :, :, :] = img


''' 미완성
if __name__ == "__main__":
    pass 
'''

train, test, train_label, test_label = load_data()
train, vaild, train_label, valid_label = train_test_split(train, train_label, test_size=0.2)
epoch = 10
batch_size = 64

for i in range(epoch):
    train, test, train_label, test_label = batch_data(train, test, train_label, test_label)

## 질문4. valid 언제 나누나??? (질문 2와 연계)