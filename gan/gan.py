import numpy as np
import tensorflow as tf
from mnistdata.loader import MNIST

Z_SIZE = 100
Z_M = 5.

def noise_batch(batch_size):
    return np.random.randn(batch_size, Z_SIZE)*Z_M

def g(z):
    with tf.variable_scope('g', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1', [Z_SIZE, 300])
        b1 = tf.get_variable('b1', [300], initializer=tf.initializers.zeros)
        w2 = tf.get_variable('w2', [300, 500])
        b2 = tf.get_variable('b2', [500], initializer=tf.initializers.zeros)
        w3 = tf.get_variable('w3', [500, 784])
        b3 = tf.get_variable('b3', [784], initializer=tf.initializers.zeros)
        
        h1 = tf.nn.relu(tf.matmul(z, w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        x_g = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)
    return x_g

def d(x):
    with tf.variable_scope('d', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1', [784, 300])
        b1 = tf.get_variable('b1', [300], initializer=tf.initializers.zeros)
        w2 = tf.get_variable('w2', [300, 100])
        b2 = tf.get_variable('b2', [100], initializer=tf.initializers.zeros)
        w3 = tf.get_variable('w3', [100, 1])
        b3 = tf.get_variable('b3', [1], initializer=tf.initializers.zeros)
        
        h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        r = tf.matmul(h2, w3) + b3 # sigmoid applied later
    return r

def model(z, x):
    x_g = g(z)
    d_real = d(x)
    d_fake = d(x_g)
    
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
    g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(d_fake), logits=d_fake
            )
    )
    return d_loss, g_loss, x_g

# define graph
x = tf.placeholder(tf.float32, [None, 784], name='x')
z = tf.placeholder(tf.float32, [None, Z_SIZE], name='z')

d_loss, g_loss, x_g = model(z, x)

all_variables = tf.trainable_variables()
g_vars = [v for v in all_variables if v.name.startswith('g')]
d_vars = [v for v in all_variables if v.name.startswith('d')]

d_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
g_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

# summary
tf.summary.scalar('g_loss', g_loss)
tf.summary.scalar('d_loss', d_loss)

tf.summary.image('x_g', tf.reshape(x_g, [-1, 28, 28, 1]), 1)

all_summary = tf.summary.merge_all()

# start session
#config = tf.ConfigProto()
#config.intra_op_parallelism_threads = 4
#config.inter_op_parallelism_threads = 4

#sess = tf.Session(config=config)
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
    z: noise_batch(batch_size)
})
writer.add_summary(summary_str, 0)

for epoch in range(1, epochs + 1):
    for batch in range(1, batches + 1):
        x_batch, _ = mnist.next_batch(batch_size)
        z_batch = noise_batch(batch_size)
        sess.run(d_opt, {x: x_batch, z: z_batch})
        
        for i in range(1):
            z_batch = noise_batch(batch_size)
            sess.run(g_opt, {z: z_batch})
        
        batch_step += 1
        print('\repoch {0} {1:3.0f} %'.format(
                epoch, batch / batches * 100), end='', flush=True
        )
    # write summary
    summary_str = sess.run(all_summary, {
        x: mnist.train_imgs,
        z: noise_batch(batch_size)
    })
    writer.add_summary(summary_str, epoch)
print('\rDone', ' '*25, flush=True)

# closing
saver.save(sess, 'model/model.ckpt')
writer.close()
sess.close()




