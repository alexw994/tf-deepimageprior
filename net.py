import tensorflow as tf
from ops import *
# 源码实验中对
# configs for super-resolution (paper default)
z_channels = 32
N = 5
nu = [128] * N
nd = [128] * N

ku = [3] * N
kd = [3] * N

ns = [4] * N
ks = [1] * N

sigma_p = 1/30
num_iter = 2000
LR = 0.01

def d_block(x, ndi, kdi, name = 'd_i'):
    with tf.variable_scope(name):
        # Downsample 使用stride = 2代替lanczos2
        x = Conv(x, ndi, kdi, stride = 2, name= 'conv1')
        # 'd_0/conv1/BiasAdd:0'
        x = BN(x, name = 'bn1')
        # 'd_0/bn/FusedBatchNorm:0'
        x = LeakyReLU(x)

        x = Conv(x, ndi, kdi, name = 'conv2')
        x = BN(x, name = 'bn2')
        x = LeakyReLU(x)
    return  x

def u_block(x, nui, kui, name = 'u_i'):
    with tf.variable_scope(name):
        x = BN(x,  name = 'bn1')

        x = Conv(x, nui, kui, name= 'conv1')
        x = BN(x,  name = 'bn2')
        x = LeakyReLU(x)

        x = Conv(x, nui, 1, name = 'conv2')
        x = BN(x)
        x = LeakyReLU(x)

        x = Upsample(x)
    return  x

def skip_block(x, nsi, ksi, name = 'skip_i'):
    with tf.variable_scope(name):
        x = Conv(x, nsi, ksi, name= 'conv1')
        x = BN(x)
        x = LeakyReLU(x)
    return  x

def forward(x, factor):
    skips = []
    for i in range(N):
        x = d_block(x, nd[i], kd[i], name = 'd_{}'.format(i))
        skip_out = skip_block(x, ns[i], ks[i], name = 's_{}'.format(i))
        skips.append(skip_out)

    out = u_block(skips[N-1], nu[N-1], ku[N-1], name='u_{}'.format(N-1))

    for i in range(N-1).__reversed__():
        out = u_block(tf.concat([out, skips[i]], axis=3),
                      nu[i], ku[i], name='u_{}'.format(i))
    # restore
    out_HR = Conv(out, 3, 1, name = 'output')
    out_LR = Downsample(out_HR, factor)

    return out_HR, out_LR


