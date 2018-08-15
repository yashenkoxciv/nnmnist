import numpy as np
import tensorflow as tf
from mnistdata.loader import MNIST

Z_SIZE = 100
C_SIZE = 10

def z_batch(batch_size):
    return np.random.randn(batch_size, Z_SIZE)*1.

def c_batch(batch_size):
    return np.random.multinomial(1, C_SIZE*[1/C_SIZE], size=batch_size)

def g(z, c):
    if hasattr(g, 'reuse'):
        g.reuse = True
    else:
        g.reuse = False
    h = tf.concat([z, c], axis=1)
    h1 = tf.layers.dense(h, 300, tf.nn.relu, name='g_h1', reuse=g.reuse)
    xg = tf.layers.dense(h1, 784, tf.nn.sigmoid, name='g_x', reuse=g.reuse)
    return xg

def dc(x):
    if hasattr(dc, 'reuse'):
        dc.reuse = True
    else:
        dc.reuse = False
    h1 = tf.layers.dense(x, 100, tf.nn.relu, name='dc_h1', reuse=dc.reuse)
    logits = tf.layers.dense(h1, 1, name='dc_l', reuse=dc.reuse)
    return logits

def q(x):
    if hasattr(q, 'reuse'):
        q.reuse = True
    else:
        q.reuse = False
    h1 = tf.layers.dense(x, 300, tf.nn.relu, name='q_h1', reuse=q.reuse)
    cr = tf.layers.dense(h1, C_SIZE, tf.nn.softmax, name='q_cr', reuse=q.reuse)
    return cr

def model(z, c, x):
    xg = g(z, c)
    q_cr = q(xg)
    dc_real = dc(x)
    dc_fake = dc(xg)
    
    cond_ent = tf.reduce_mean(
            -tf.reduce_sum(tf.log(q_cr + 1e-8)*c, 1)
    )
    q_loss = cond_ent
    
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
    g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(dc_fake), logits=dc_fake
            )
    )
    return dc_loss, g_loss, q_loss

if __name__ == '__main__':
    # define graph
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    z = tf.placeholder(tf.float32, [None, Z_SIZE], name='z')
    c = tf.placeholder(tf.float32, [None, C_SIZE], name='c')
    
    dc_loss, g_loss, q_loss = model(z, c, x)
    
    all_variables = tf.trainable_variables()
    g_vars = [v for v in all_variables if v.name.startswith('g_')]
    dc_vars = [v for v in all_variables if v.name.startswith('dc_')]
    q_vars = [v for v in all_variables if v.name.startswith('q_')]
    qg_vars = g_vars + q_vars
    
    dc_opt = tf.train.AdamOptimizer().minimize(dc_loss, var_list=dc_vars)
    g_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)
    qg_opt = tf.train.AdamOptimizer().minimize(q_loss, var_list=qg_vars)
    
    # summary
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('dc_loss', dc_loss)
    tf.summary.scalar('q_loss', q_loss)
    tf.summary.image('x_g', tf.reshape(g(z, c), [-1, 28, 28, 1]), 1)
    all_summary = tf.summary.merge_all()
    
    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary/', sess.graph)
    saver = tf.train.Saver()
    
    # load data
    mnist = MNIST('../MNIST')
    
    # set train parameters
    epochs = 50
    batch_size = 100
    batches = mnist.train_imgs.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, _ = mnist.next_batch(batch_size)
            
            sess.run(dc_opt, {
                    x: x_batch,
                    z: z_batch(batch_size),
                    c: c_batch(batch_size)
            })
            sess.run(g_opt, {
                    z: z_batch(batch_size),
                    c: c_batch(batch_size)
            })
            sess.run(qg_opt, {
                    z: z_batch(batch_size),
                    c: c_batch(batch_size)
            })
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100), end='', flush=True
            )
        # write summary
        summary_str = sess.run(all_summary, {
            x: x_batch,
            z: z_batch(batch_size),
            c: c_batch(batch_size)
        })
        writer.add_summary(summary_str, epoch-1)
    print('\rDone', ' '*25, flush=True)

    # closing
    saver.save(sess, 'model/model.ckpt')
    writer.close()
    sess.close()
    
    