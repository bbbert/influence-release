from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import datasets as ds
import datasets.mnist
import datasets.cifar10
from experiments.common import Experiment, phase, collect_phases

import time
import numpy as np
import tensorflow as tf
import tensorflow.data

def check_alignment(dataset, sample_weights, restore=False):
    tf.reset_default_graph()
    sess = tf.Session()

    x_placeholder = tf.placeholder(dataset.x.dtype, dataset.x.shape)
    labels_placeholder = tf.placeholder(dataset.labels.dtype, dataset.labels.shape)
    sample_weights_placeholder = tf.placeholder(sample_weights.dtype, sample_weights.shape)
    train = tf.data.Dataset.from_tensor_slices({
        'x': x_placeholder,
        'label': labels_placeholder,
        'sample_weight': sample_weights_placeholder,
    })

    K = 10

    batched = train.batch(K)#.shuffle(142, seed=3042034)
    iterator = batched.make_initializable_iterator()
    saveable = tf.data.experimental.make_saveable_from_iterator(iterator)
    f = tf.Variable(3, dtype=tf.float32)

    saver = tf.train.Saver([f, saveable])
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={
        x_placeholder: dataset.x,
        labels_placeholder: dataset.labels,
        sample_weights_placeholder: sample_weights,
    })
    next_op = iterator.get_next()

    if restore:
        saver.restore(sess, 'output/blah')
    else:
        # sess.run(iterator.initializer, feed_dict={
        #     x_placeholder: dataset.x,
        #     labels_placeholder: dataset.labels,
        #     sample_weights_placeholder: sample_weights,
        # })
        sess.run(next_op)
        sess.run(next_op)
        sess.run(next_op)
        sess.run(next_op)
        sess.run(next_op)
        sess.run(iterator.initializer, feed_dict={
            x_placeholder: dataset.x,
            labels_placeholder: dataset.labels,
            sample_weights_placeholder: sample_weights,
        })
        sess.run(next_op)
        sess.run(next_op)
        sess.run(next_op)
        saver.save(sess, 'output/blah', global_step=34)

    state = tf.train.get_checkpoint_state('output', latest_filename='checkpoint')
    print(state.all_model_checkpoint_paths)

    for i in range(10):
        elem = sess.run(next_op)
        new_x = elem['x']
        new_label = elem['label']
        new_sample_weight = elem['sample_weight']
        old_x = dataset.x[(i+3)*K:(i+4)*K, :]
        old_label = dataset.labels[(i+3)*K:(i+4)*K]
        print(np.linalg.norm(new_x - old_x), new_label, old_label, new_sample_weight)

    sess.close()

def index_by_tf(dataset, indices):
    tf.reset_default_graph()
    sess = tf.Session()

    x_placeholder = tf.placeholder(tf.float32, (None, dataset.x.shape[1]))
    indices_placeholder = tf.placeholder(tf.int32, (None,))

    v = tf.get_variable('v',
                        dtype=tf.float32,
                        shape=(3,),
                        initializer=tf.constant_initializer(-3))
    w = tf.get_variable('w',
                        dtype=tf.float32,
                        shape=(6,),
                        initializer=tf.constant_initializer(-4))
    vw = tf.concat([tf.reshape(x, (-1,)) for x in [v, w]], axis=0)
    v_ = vw[:v.shape[0]]
    w_ = vw[v.shape[0]:]

    # this doesn't work: tf can't tell v = v_
    #p = tf.tensordot(v, w[:3], 0) + tf.tensordot(v, w[3:], 0) ** 2

    # this does
    p = tf.tensordot(v_, w_[:3], 0) + tf.tensordot(v_, w_[3:], 0) ** 2

    xs = tf.gather(x_placeholder, indices_placeholder)
    sums = tf.reduce_sum(xs, axis=1)
    ans = tf.reduce_sum(sums, axis=0)

    sess.run(tf.global_variables_initializer())
    print(sess.run(vw))
    print(sess.run(v_))
    print(sess.run(w_))

    hessp = tf.hessians(p, [v, w])
    hessp_ = tf.hessians(p, [v_, w_])
    hesspa = tf.hessians(p, vw)
    print(sess.run(p))
    print(sess.run(hessp))
    print(sess.run(hessp_))
    print(sess.run(hesspa))

    result = sess.run(ans, feed_dict={
        x_placeholder: dataset.x,
        indices_placeholder: indices,
    })
    return result

def index_by_np(dataset, indices):
    tf.reset_default_graph()
    sess = tf.Session()

    x_placeholder = tf.placeholder(tf.float32, (None, dataset.x.shape[1]))

    xs = x_placeholder
    sums = tf.reduce_sum(xs, axis=1)
    ans = tf.reduce_sum(sums, axis=0)

    sess.run(tf.global_variables_initializer())

    result = sess.run(ans, feed_dict={
        x_placeholder: dataset.x[indices, :],
    })
    return result

if __name__ == "__main__":
    datasets = ds.mnist.load_small_mnist()
    dataset = datasets.train
    sample_weights = np.ones((dataset.x.shape[0],))
    #check_alignment(dataset, sample_weights, restore=False)
    #check_alignment(dataset, sample_weights, restore=True)

    print(dataset.x.shape)
    indices = np.random.choice(dataset.x.shape[0], dataset.x.shape[0] - 100, replace=False)

    time_st = time.time()
    result = index_by_tf(dataset, indices)
    dur = time.time() - time_st
    print(result, dur)

    time_st = time.time()
    result = index_by_np(dataset, indices)
    dur = time.time() - time_st
    print(result, dur)

