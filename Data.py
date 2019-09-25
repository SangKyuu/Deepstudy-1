import os
from glob import glob
import random
import numpy as np
from PIL import Image

def Ramen_Dataset():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(root_dir, 'dataset', 'train', '*\\*.png')
    test_path = os.path.join(root_dir, 'dataset', 'val', '*\\*.png')

    train_file = glob(train_path)
    test_file = glob(test_path)
    return train_file, test_file

def read_image(path):
    image = Image.open(path)
    return np.asarray(image.resize((600, 600)))

def DataLoader(files, batch_size=32):
    random.shuffle(files)
    for i in range(0, len(files), batch_size):
        batch_data = np.array(files[i:i + batch_size])
        batch_labels = np.zeros(batch_size)
        batch_images = np.zeros((batch_size, 600, 600, 3))
        #batch_labelf = np.core.defchararray.split(batch_data, sep='\\')
        for n, file in enumerate(batch_data):
            batch_images[n,:,:,:] = read_image(file)
            batch_labels[n] = file.split('\\')[-2]
        yield batch_images, batch_labels

tr, te = Ramen_Dataset()
train_Loader = DataLoader(tr, 32)
for batch in train_Loader:
    print(batch)
