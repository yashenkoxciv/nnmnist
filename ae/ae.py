import tensorflow as tf
from mnistdata.loader import MNIST

Z_SIZE = 100

def e(x):
    with tf.variable_scope('e', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1', [784, 300])
        b1 = tf.get_variable('b1', [300], initializer=tf.initializers.zeros)
        w2 = tf.get_variable('w2', [300, Z_SIZE])
        b2 = tf.get_variable('b2', [Z_SIZE], initializer=tf.initializers.zeros)
        
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    z = tf.matmul(h1, w2) + b2
    return z

def d(z):
    with tf.variable_scope('d', reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable('w1', [Z_SIZE, 300])
        b1 = tf.get_variable('b1', [300], initializer=tf.initializers.zeros)
        w2 = tf.get_variable('w2', [300, 784])
        b2 = tf.get_variable('b2', [784], initializer=tf.initializers.zeros)
        
    h1 = tf.nn.relu(tf.matmul(z, w1) + b1)
    x_r = tf.nn.sigmoid(tf.matmul(h1, w2) + b2)
    return x_r

def model(x):
    z = e(x)
    x_r = d(z)
    
    loss = tf.reduce_mean(tf.square(x - x_r))
    return loss, x_r

if __name__ == '__main__':
    # define graph
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    test_x = tf.placeholder(tf.float32, [None, 784], name='x')
    
    loss, _ = model(x)
    test_loss, x_r = model(test_x)
    
    opt = tf.train.AdamOptimizer().minimize(loss)
    
    # summary
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('test_loss', test_loss)
    tf.summary.image('x_r', tf.reshape(x_r, [-1, 28, 28, 1]), 1)
    
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
        test_x: mnist.test_imgs
    })
    writer.add_summary(summary_str, 0)
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, _ = mnist.next_batch(batch_size)
            
            sess.run(opt, {x: x_batch})
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100), end='', flush=True
            )
        # write summary
        summary_str = sess.run(all_summary, {
            x: mnist.train_imgs,
            test_x: mnist.test_imgs
        })
        writer.add_summary(summary_str, epoch)
    print('\rDone', ' '*25, flush=True)
    
    # closing
    saver.save(sess, 'model/model.ckpt')
    writer.close()
    sess.close()
    
    
    
    
    