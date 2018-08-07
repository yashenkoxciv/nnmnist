import numpy as np
import tensorflow as tf
from mnistdata.loader import MNIST

Z_SIZE = 100

def z_batch(batch_size, m=5):
    return np.random.randn(batch_size, Z_SIZE)*m

def e(x):
    with tf.variable_scope('e', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1', [784, 300])
        b1 = tf.get_variable('b1', [300], initializer=tf.initializers.zeros)
        w2 = tf.get_variable('w2', [300, Z_SIZE])
        b2 = tf.get_variable('b2', [Z_SIZE], initializer=tf.initializers.zeros)
        
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    z = tf.matmul(h1, w2) + b2
    return z

def g(z):
    with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1', [Z_SIZE, 300])
        b1 = tf.get_variable('b1', [300], initializer=tf.initializers.zeros)
        w2 = tf.get_variable('w2', [300, 784])
        b2 = tf.get_variable('b2', [784], initializer=tf.initializers.zeros)
        
    h1 = tf.nn.relu(tf.matmul(z, w1) + b1)
    x_r = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)
    return x_r

def d(z):
    with tf.variable_scope('d', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1', [Z_SIZE, 100])
        b1 = tf.get_variable('b1', [100], initializer=tf.initializers.zeros)
        w2 = tf.get_variable('w2', [100, 1])
        b2 = tf.get_variable('b2', [1], initializer=tf.initializers.zeros)
        
    h1 = tf.nn.relu(tf.matmul(z, w1) + b1)
    logits = tf.matmul(h1, w2) + b2
    return logits

def model(x, pz):
    z = e(x)
    x_r = g(z)
    d_fake = d(z)
    d_real = d(pz)
    
    d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_real), logits=d_real
            )
    )
    d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(d_fake), logits=d_fake
            )
    )
    d_loss = d_real_loss + d_fake_loss
    
    e_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_fake), logits=d_fake
            )
    )
    
    r_loss = tf.reduce_mean(tf.square(x - x_r))
    return r_loss, e_loss, d_loss, x_r, z

if __name__ == '__main__':
    # define graph
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    z = tf.placeholder(tf.float32, [None, Z_SIZE], name='z')
    
    test_x = tf.placeholder(tf.float32, [None, 784], name='test_x')
    test_z = tf.placeholder(tf.float32, [None, Z_SIZE], name='test_z')
    
    r_loss, e_loss, d_loss, _, _ = model(x, z)
    test_r_loss, test_e_loss, test_d_loss, x_r, x_z = model(test_x, test_z)
    
    all_variables = tf.trainable_variables()
    e_vars = [v for v in all_variables if v.name.startswith('e')]
    d_vars = [v for v in all_variables if v.name.startswith('d')]
    
    r_opt = tf.train.AdamOptimizer().minimize(r_loss)
    d_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
    e_opt = tf.train.AdamOptimizer().minimize(e_loss, var_list=e_vars)
    
    # summary
    tf.summary.scalar('r_loss', r_loss)
    tf.summary.scalar('e_loss', e_loss)
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('test_r_loss', test_r_loss)
    tf.summary.scalar('test_e_loss', test_e_loss)
    tf.summary.scalar('test_d_loss', test_d_loss)
    tf.summary.image('x_r', tf.reshape(x_r, [-1, 28, 28, 1]), 1)
    tf.summary.histogram('x_z', x_z)
    tf.summary.histogram('z', test_z)
    
    all_summary = tf.summary.merge_all()
    
    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary/', sess.graph)
    saver = tf.train.Saver()
    
    # load data
    mnist = MNIST('../MNIST')
    
    # set train parameters
    epochs = 20
    batch_size = 10
    batches = mnist.train_imgs.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    
    # summary before learning
    summary_str = sess.run(all_summary, {
        x: mnist.train_imgs,
        test_x: mnist.test_imgs,
        z: z_batch(mnist.train_imgs.shape[0]),
        test_z: z_batch(mnist.test_imgs.shape[0])
    })
    writer.add_summary(summary_str, 0)
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, _ = mnist.next_batch(batch_size)
            
            sess.run(r_opt, {x: x_batch})
            sess.run(d_opt, {x: x_batch, z: z_batch(batch_size)})
            sess.run(e_opt, {x: x_batch})
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100), end='', flush=True
            )
        # write summary
        summary_str = sess.run(all_summary, {
            x: mnist.train_imgs,
            test_x: mnist.test_imgs,
            z: z_batch(mnist.train_imgs.shape[0]),
            test_z: z_batch(mnist.test_imgs.shape[0])
        })
        writer.add_summary(summary_str, epoch)
    print('\rDone', ' '*25, flush=True)
    
    # closing
    saver.save(sess, 'model/model.ckpt')
    writer.close()
    sess.close()
    
    
    
    
    