import tensorflow as tf
from mnistdata.loader import MNIST

def n(x):
    if hasattr(n, 'reuse'):
        n.reuse = True
    else:
        n.reuse = False
    c1 = tf.layers.conv2d(
            x, filters=32, kernel_size=[5, 5],
            strides=[2, 2], padding='same',
            activation=tf.nn.relu,
            name='c1', reuse=n.reuse
    )
    p1 = tf.layers.max_pooling2d(
            c1, pool_size=[10, 10], strides=[2, 2],
            padding='same', name='p1'
    )
    c2 = tf.layers.conv2d(
            p1, filters=64, kernel_size=[5, 5],
            strides=[2, 2], padding='same',
            activation=tf.nn.relu,
            name='c2', reuse=n.reuse
    )
    p2 = tf.layers.max_pooling2d(
            c2, pool_size=[5, 5], strides=[2, 2],
            padding='same', name='p2'
    )
    cf = tf.layers.flatten(p2, name='flatten1')
    d1 = tf.layers.dense(
            cf, units=100, activation=tf.nn.relu,
            name='d1', reuse=n.reuse
    )
    logits = tf.layers.dense(
            d1, units=10,
            name='d2', reuse=n.reuse
    )
    #print(c1, p1, c2, p2, cf, d1, logits, sep='\n')
    return logits

def model(x, y):
    xs = tf.reshape(x, [-1, 28, 28, 1])
    y1h = tf.one_hot(y, 10, 1, 0)
    
    logits = n(xs)
    
    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits,
                    labels=y1h
            )
    )
    
    pr_y = tf.nn.softmax(logits)
    pr_labels = tf.argmax(pr_y, axis=1) # , output_type=tf.int32
    correctness = tf.equal(pr_labels, y)
    accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))
    return loss, accuracy

if __name__ == '__main__':
    # define graph
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y = tf.placeholder(tf.int64, [None], name='y')
    
    test_x = tf.placeholder(tf.float32, [None, 784], name='x')
    test_y = tf.placeholder(tf.int64, [None], name='y')
    
    loss, accuracy = model(x, y)
    test_loss, test_accuracy = model(test_x, test_y)
    
    opt = tf.train.AdamOptimizer().minimize(loss)
    
    # summary
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('test_loss', test_loss)
    tf.summary.scalar('test_accuracy', test_accuracy)
    
    all_summary = tf.summary.merge_all()
    
    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary/', sess.graph)
    saver = tf.train.Saver()
    
    # load data
    mnist = MNIST('../MNIST')
    
    # set train parameters
    epochs = 12
    batch_size = 100
    batches = mnist.train_imgs.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    
    # summary before learning
    #summary_str = sess.run(all_summary, {
    #    x: mnist.train_imgs,
    #    y: mnist.train_labels,
    #    test_x: mnist.test_imgs,
    #    test_y: mnist.test_labels
    #})
    #writer.add_summary(summary_str, 0)
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, y_batch = mnist.next_batch(batch_size)
            
            sess.run(opt, {x: x_batch, y: y_batch})
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100), end='', flush=True
            )
            # write summary
            if batch_step % 300 == 0:
                summary_str = sess.run(all_summary, {
                    x: x_batch,
                    y: y_batch,
                    test_x: mnist.test_imgs[:1000],
                    test_y: mnist.test_labels[:1000]
                })
                writer.add_summary(summary_str, batch_step)
    print('\rDone', ' '*25, flush=True)
    
    # closing
    saver.save(sess, 'model/model.ckpt')
    writer.close()
    sess.close()

