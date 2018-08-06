import tensorflow as tf
from mnistdata.loader import MNIST

def n(x):
    with tf.variable_scope('n', reuse=tf.AUTO_REUSE):
        w = tf.get_variable(
                'w',
                [784, 10],
                initializer=tf.initializers.zeros
        )
        b = tf.get_variable(
                'b',
                [10],
                initializer=tf.initializers.zeros
        )
        logits = tf.matmul(x, w) + b # apply softmax later
    return logits

def model(x, y):
    y1h = tf.one_hot(y, 10, 1, 0)
    
    logits = n(x)
    
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
    batch_size = 10
    batches = mnist.train_imgs.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    
    # summary before learning
    summary_str = sess.run(all_summary, {
        x: mnist.train_imgs,
        y: mnist.train_labels,
        test_x: mnist.test_imgs,
        test_y: mnist.test_labels
    })
    writer.add_summary(summary_str, 0)
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, y_batch = mnist.next_batch(batch_size)
            
            sess.run(opt, {x: x_batch, y: y_batch})
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100), end='', flush=True
            )
        # write summary
        summary_str = sess.run(all_summary, {
            x: mnist.train_imgs,
            y: mnist.train_labels,
            test_x: mnist.test_imgs,
            test_y: mnist.test_labels
        })
        writer.add_summary(summary_str, epoch)
    print('\rDone', ' '*25, flush=True)
    
    # closing
    saver.save(sess, 'model/model.ckpt')
    writer.close()
    sess.close()
