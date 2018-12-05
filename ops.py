import tensorflow as tf
import scipy.stats as st
import numpy as np

def Conv(x, filters, kernel_size, stride = 1, name= 'conv'):
    # reflection padding tensorflow 中未实现，代替使用0补齐
    return tf.layers.conv2d(x, filters, kernel_size,
                            strides = stride,
                            padding = 'same',
                            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-4),
                            name = name)

def LeakyReLU(x):
    return tf.nn.leaky_relu(x)


def BN(x, name = 'bn'):
    return tf.layers.batch_normalization(x, training = True, name = name)

def Upsample(x):
    # default 'bilinear'， method = 0
    height, width = x.get_shape()[1: 3]
    x = tf.image.resize_images(x, [height * 2, width * 2], method = 0)
    return x

def Downsample(x, factor = 2):
    # default 'lanczos2'
    # downsample by conv
    # 先使用Gaussian kernel代替
    Gaussian_kernel  = gkern()
    gaus_filter = tf.expand_dims(tf.stack([Gaussian_kernel, Gaussian_kernel, Gaussian_kernel], axis=2), axis=3)
    return tf.nn.depthwise_conv2d(x, gaus_filter, strides=[1, factor, factor, 1], padding='SAME')

def gkern(kernlen=5, nsig=3):

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return tf.convert_to_tensor(kernel, dtype=tf.float32)


