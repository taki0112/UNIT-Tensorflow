import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np


class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image):
        if self.pool_size == 0:
            return image

        if len(self.images) < self.pool_size:
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image

class ImageData:

    def __init__(self, batch_size, load_size, channels, augment_flag):
        self.batch_size = batch_size
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag

    def image_processing(self, filename):
        x = tf.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            augment_size = self.load_size + (30 if self.load_size == 256 else 15)
            p = random.random()
            if p > 0.5:
                img = augmentation(img, augment_size)

        return img



def prepare_data(dataset_name, size):
    data_path = os.path.join("./dataset", dataset_name)

    trainA = []
    trainB = []
    for path, dir, files in os.walk(data_path):
        for file in files:
            image = os.path.join(path, file)
            if path.__contains__('trainA') :
                trainA.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))
            if path.__contains__('trainB') :
                trainB.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))


    trainA = preprocessing(np.asarray(trainA))
    trainB = preprocessing(np.asarray(trainB))

    np.random.shuffle(trainA)
    np.random.shuffle(trainB)

    return trainA, trainB

def test_data(dataset_name, size) :
    data_path = os.path.join("./dataset", dataset_name)
    testA = []
    testB = []
    for path, dir, files in os.walk(data_path) :
        for file in files :
            image = os.path.join(path, file)
            if path.__contains__('testA') :
                testA.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))
            if path.__contains__('testB') :
                testB.append(misc.imresize(misc.imread(image, mode='RGB'), [size, size]))

    testA = preprocessing(np.asarray(testA))
    testB = preprocessing(np.asarray(testB))

    return testA, testB

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    """
    # Create Normal distribution
    x = x.astype('float32')
    x[:, :, :, 0] = (x[:, :, :, 0] - np.mean(x[:, :, :, 0])) / np.std(x[:, :, :, 0])
    x[:, :, :, 1] = (x[:, :, :, 1] - np.mean(x[:, :, :, 1])) / np.std(x[:, :, :, 1])
    x[:, :, :, 2] = (x[:, :, :, 2] - np.mean(x[:, :, :, 2])) / np.std(x[:, :, :, 2])
    """
    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation(image, augment_size):
    seed = random.randint(0, 2 ** 31 - 1)
    ori_image_shape = tf.shape(image)
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.resize_images(image, [augment_size, augment_size])
    image = tf.random_crop(image, ori_image_shape, seed=seed)
    return image

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir