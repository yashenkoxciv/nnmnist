import os
import gzip
import shutil
import logging
import numpy as np
from urllib.request import urlopen

default_urls = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]

default_files = [f.split('/')[-1] for f in default_urls]

def download_file(url, file_name):
    with urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

def download(mnist_dir):
    logging.info('Downloading MNIST')
    
    for i in range(len(default_urls)):
        download_file(
                default_urls[i],
                os.path.join(mnist_dir, default_files[i])
        )
        logging.info(default_files[i] + ' downloaded')
    
def byte2int(b):
    return int.from_bytes(b, byteorder='big')

def load_file(file_name):
    with gzip.open(file_name, 'rb') as f:
        magic_number = byte2int(f.read(4))
        items_count = byte2int(f.read(4))
        if 'images' in file_name:
            img_r = byte2int(f.read(4))
            img_c = byte2int(f.read(4))
            img_len = img_r*img_c
            imgs = list(f.read(img_len*items_count))
            return imgs
        elif 'labels' in file_name:
            labels = list(f.read(items_count))
            return labels
        else:
            raise Exception('Can\'t recognize file by name')
        

def load(mnist_dir='MNIST'):
    if not os.path.isdir(mnist_dir):
        os.mkdir(mnist_dir)
    # if directory contains only MNIST's files
    if set(os.listdir(mnist_dir)) == set(default_files):
        pass
    elif set(os.listdir(mnist_dir)) & set(default_files) == set(default_files):
        logging.warning('MNIST directory contain some other files')
    else:
        logging.info('Can\'t find MNIST files')
        download(mnist_dir)
    train_imgs_file = os.path.join(mnist_dir, default_files[0])
    train_imgs = load_file(train_imgs_file)
    
    train_labels_file = os.path.join(mnist_dir, default_files[1])
    train_labels = load_file(train_labels_file)
    
    test_imgs_file = os.path.join(mnist_dir, default_files[2])
    test_imgs = load_file(test_imgs_file)
    
    test_labels_file = os.path.join(mnist_dir, default_files[3])
    test_labels = load_file(test_labels_file)
    
    logging.info('MNIST loaded')
    return train_imgs, train_labels, test_imgs, test_labels

class MNIST:
    def __init__(self, mnist_dir='MNIST'):
        train_imgs, train_labels, test_imgs, test_labels = load(mnist_dir)
        self.train_imgs = np.float32(train_imgs).reshape([-1, 784]) / 256
        self.test_imgs = np.float32(test_imgs).reshape([-1, 784]) / 256
        self.train_labels = np.uint8(train_labels)
        self.test_labels = np.uint8(test_labels)
        self.train_size = self.train_imgs.shape[0]
    
    def next_batch(self, batch_size):
        batch_idx = np.random.choice(self.train_size, batch_size)
        batch_imgs = self.train_imgs[batch_idx]
        batch_labels = self.train_labels[batch_idx]
        return batch_imgs, batch_labels
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    logging.basicConfig(level=logging.INFO)
    mnist = MNIST()
    N = 5
    imgs, labels = mnist.next_batch(N)
    print(imgs.max(), imgs.min())
    for i in range(N):
        plt.imshow(imgs[i].reshape([28, 28]), cmap='gray')
        plt.title(labels[i])
        plt.show()
    
