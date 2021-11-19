import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import pandas as pd

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train.astype('uint8'), x_test.astype('uint8')
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, shuffle=False, random_state=456)
    return x_train, x_test, y_train, y_test

def transform(image):
    img = cv2.resize(image, (28,28))
    img = img / 255
    img = np.expand_dims(img, axis=-1)
    return img

def add_random_noise(img,x1,x2,y1,y2):
    white_noise = np.random.uniform(0,1,size=(x2-x1,y2-y1))
    img[x1:x2,y1:y2] = white_noise
    return img

def prepare_data(images,image_labels):
    '''
    ### if batch size = 64
    :param images: images is mnist data, shape = (64,28,28,1)
    :param image_label: image label, shape = (64,)
    :return: occ_image, nonocc_image, mnist_label
    '''

    occ_image_list, nonocc_image_list, mnist_label_list = [], [], []
    label_data = [[] for _ in range(10)] ## 10 class

    for idx in range(images.shape[0]):
        image,label = images[idx], image_labels[idx]
        label_data[label].append(image)

    x1,x2,y1,y2 = 0,7,0,7
    for i in range(10):
        data = label_data[i]
        for data_idx in range(len(data)):
            random_choose_idx = random.randint(0,len(data)-1)
            Nonocc_image = data[data_idx]
            Occ_image = data[random_choose_idx]
            Occ_image = add_random_noise(Occ_image, x1, x2, y1, y2)
            label = i
            nonocc_image_list.append(Nonocc_image)
            occ_image_list.append(Occ_image), mnist_label_list.append(label)

    mnist_label_list = pd.get_dummies(mnist_label_list)
    return np.expand_dims(np.array(occ_image_list),axis=-1), np.expand_dims(np.array(nonocc_image_list),axis=-1), np.array(mnist_label_list)