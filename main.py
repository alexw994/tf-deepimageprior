import tensorflow as tf
import matplotlib.pylab as plt
from net import *
from utils import *
from PIL import Image

image_name = 'zebra_crop'
width = 256
height = 256
factor = 4

val_HR, val_LR = load_img(image_name + '.png', height, width, factor)

random_z = tf.placeholder(tf.float32, [1, height, width, 32])
out_HR, out_LR = forward(random_z, factor)

loss = tf.losses.mean_squared_error(val_LR, out_LR)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

save_image('{}_reduce{}.png'.format(image_name, factor), tf.squeeze(val_LR))
val_LR_pil = Image.fromarray(sess.run(tf.squeeze(val_LR)))
val_HR_pil = Image.fromarray(sess.run(tf.squeeze(val_HR)))

plt.imsave('{}_BICUBIC.png'.format(image_name), np.asarray(val_LR_pil.resize(val_HR_pil.size, Image.BICUBIC)))
import os
if not os.path.exists('output'):
    os.makedirs('output')

if not os.path.exists('model'):
    os.makedirs('model')

for i in range(5000  + 1):
    new_rand = np.random.uniform(0, 1.0/10.0, size=(1, height, width, 32)).astype(np.float32)
    new_rand = new_rand + np.random.rand(1, height, width, 32) * (1/30)
    _, lossval = sess.run(
        [train_op, loss],
        feed_dict = {random_z: new_rand}
    )
    if i % 100 == 0:
        image_out = np.squeeze(sess.run(out_HR, feed_dict={random_z: new_rand}))
        plt.imsave("output/%d_%s" % (i, image_name), image_out)
    print(i, lossval)
    if i % 500 == 0:
        model = {}
        for j in tf.trainable_variables():
            model[j.name] = sess.run(j)
        np.save('model\{}_model'.format(i), model)
