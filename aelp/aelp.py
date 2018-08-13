import numpy as np
import tensorflow as tf
from mnistdata.loader import MNIST

P_DIM = 50
Z_DIM = 100

def prior(batch_size):
    return np.random.randn(batch_size, P_DIM)*2

def e(x):
    if hasattr(e, 'reuse'):
        e.reuse = True
    else:
        e.reuse = False
    c1 = tf.layers.conv2d(
            x, filters=32, kernel_size=[5, 5],
            strides=[2, 2], padding='same',
            activation=tf.nn.relu,
            name='e_c1', reuse=e.reuse
    )
    p1 = tf.layers.max_pooling2d(
            c1, pool_size=[10, 10], strides=[2, 2],
            padding='same', name='e_p1'
    )
    c2 = tf.layers.conv2d(
            p1, filters=64, kernel_size=[5, 5],
            strides=[2, 2], padding='same',
            activation=tf.nn.relu,
            name='e_c2', reuse=e.reuse
    )
    p2 = tf.layers.max_pooling2d(
            c2, pool_size=[5, 5], strides=[2, 2],
            padding='same', name='e_p2'
    )
    cf = tf.layers.flatten(p2, name='e_flat1')
    d1 = tf.layers.dense(
            cf, units=100, activation=tf.nn.relu,
            name='e_d1', reuse=e.reuse
    )
    z = tf.layers.dense(
            d1, units=Z_DIM,
            name='e_d2', reuse=e.reuse
    )
    return z

def d(z):
    if hasattr(d, 'reuse'):
        d.reuse = True
    else:
        d.reuse = False
    zs = tf.reshape(z, [-1, 5, 5, 4])
    us1 = tf.image.resize_images(
            zs, size=(7, 7),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    c1 = tf.layers.conv2d(
            us1, filters=32, kernel_size=[5, 5],
            strides=[2, 2], padding='same',
            activation=tf.nn.relu,
            name='d_c1', reuse=d.reuse
    )
    us2 = tf.image.resize_images(
            c1, size=(14, 14),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    c2 = tf.layers.conv2d(
            us2, filters=16, kernel_size=[5, 5],
            strides=[2, 2], padding='same',
            activation=tf.nn.relu,
            name='d_c2', reuse=d.reuse
    )
    us3 = tf.image.resize_images(
            c2, size=(28, 28),
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    x_r = tf.layers.conv2d(
            us3, filters=1, kernel_size=[5, 5],
            strides=[1, 1], padding='same',
            activation=tf.nn.sigmoid,
            name='d_c3', reuse=d.reuse
    )
    return x_r

def c(p):
    if hasattr(c, 'reuse'):
        c.reuse = True
    else:
        c.reuse = False
    d1 = tf.layers.dense(
            p, units=200, activation=tf.nn.relu,
            name='c_d1', reuse=c.reuse
    )
    zh = tf.layers.dense(
            d1, units=Z_DIM,
            name='c_d2', reuse=c.reuse
    )
    return zh

def dc(x):
    if hasattr(dc, 'reuse'):
        dc.reuse = True
    else:
        dc.reuse = False
    c1 = tf.layers.conv2d(
            x, filters=32, kernel_size=[5, 5],
            strides=[2, 2], padding='same',
            activation=tf.nn.relu,
            name='dc_c1', reuse=dc.reuse
    )
    p1 = tf.layers.max_pooling2d(
            c1, pool_size=[10, 10], strides=[2, 2],
            padding='same', name='dc_p1'
    )
    c2 = tf.layers.conv2d(
            p1, filters=64, kernel_size=[5, 5],
            strides=[2, 2], padding='same',
            activation=tf.nn.relu,
            name='dc_c2', reuse=dc.reuse
    )
    p2 = tf.layers.max_pooling2d(
            c2, pool_size=[5, 5], strides=[2, 2],
            padding='same', name='dc_p2'
    )
    cf = tf.layers.flatten(p2, name='dc_flat1')
    d1 = tf.layers.dense(
            cf, units=100, activation=tf.nn.relu,
            name='dc_d1', reuse=dc.reuse
    )
    logits = tf.layers.dense(
            d1, units=1,
            name='dc_d2', reuse=dc.reuse
    )
    return logits

def model(x, p):
    x_r = d(e(x))
    x_g = d(c(p))
    dc_real = dc(x_r)
    dc_fake = dc(x_g)
    
    ae_loss = tf.reduce_mean(tf.square(x - x_r))
    dc_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(dc_real), logits=dc_real
            )
    )
    dc_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(dc_fake), logits=dc_fake
            )
    )
    dc_loss = dc_real_loss + dc_fake_loss
    c_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(dc_fake), logits=dc_fake
            )
    )
    return ae_loss, dc_loss, c_loss

if __name__ == '__main__':
    # define graph
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
    p = tf.placeholder(tf.float32, [None, P_DIM], name='p')
    
    ae_loss, dc_loss, c_loss = model(x, p)
    
    all_variables = tf.trainable_variables()
    c_vars = [v for v in all_variables if v.name.startswith('c_')]
    dc_vars = [v for v in all_variables if v.name.startswith('dc_')]
    
    ae_opt = tf.train.AdamOptimizer().minimize(ae_loss)
    dc_opt = tf.train.AdamOptimizer().minimize(dc_loss, var_list=dc_vars)
    c_opt = tf.train.AdamOptimizer().minimize(c_loss, var_list=c_vars)
    
    # summary
    tf.summary.scalar('ae_loss', ae_loss)
    tf.summary.scalar('dc_loss', dc_loss)
    tf.summary.scalar('c_loss', c_loss)
    
    all_summary = tf.summary.merge_all()
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary/', sess.graph)
    saver = tf.train.Saver()
    
    # load data
    mnist = MNIST('../MNIST')
    mnist.train_imgs = mnist.train_imgs.reshape([-1, 28, 28, 1])
    
    # set train parameters
    epochs = 7
    batch_size = 100
    batches = mnist.train_imgs.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, _ = mnist.next_batch(batch_size)
            
            sess.run(ae_opt, {x: x_batch})
            sess.run(dc_opt, {x: x_batch, p: prior(batch_size)})
            sess.run(c_opt, {p: prior(batch_size)})
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100), end='', flush=True
            )
                    
            # write summary
            if batch_step % 300 == 0:
                summary_str = sess.run(all_summary, {
                    x: x_batch,
                    p: prior(batch_size)
                })
                writer.add_summary(summary_str, batch_step)
    
    print('\rDone', ' '*25, flush=True)
        
    # closing
    saver.save(sess, 'model/model.ckpt')
    writer.close()
    sess.close()




















