import tensorflow as tf
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers import variance_scaling_initializer as he_init

def conv(x, channels, kernel=3, stride=2, pad=0, normal_weight_init=False, activation_fn='leaky', scope='conv_0') :
    with tf.variable_scope(scope) :
        x = tf.pad(x, [[0,0], [pad, pad], [pad, pad], [0,0]])

        if normal_weight_init :
            x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 strides=stride, kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

        else :
            if activation_fn == 'relu' :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(), strides=stride,
                                     kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, strides=stride,
                                     kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))


        x = activation(x, activation_fn)

        return x

def deconv(x, channels, kernel=3, stride=2, normal_weight_init=False, activation_fn='leaky', scope='deconv_0') :
    with tf.variable_scope(scope):
        if normal_weight_init:
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 strides=stride, padding='SAME', kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

        else:
            if activation_fn == 'relu' :
                x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(), strides=stride, padding='SAME',
                                               kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.conv2d_transpose(inputs=x, filters=channels, kernel_size=kernel, strides=stride, padding='SAME',
                                               kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

        x = activation(x, activation_fn)

        return x

def resblock(x_init, channels, kernel=3, stride=1, pad=1, dropout_ratio=0.0, normal_weight_init=False, is_training=True, norm_fn='instance', scope='resblock_0') :
    assert norm_fn in ['instance', 'batch', 'weight', 'spectral', None]
    with tf.variable_scope(scope) :
        with tf.variable_scope('res1') :
            x = tf.pad(x_init, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

            if normal_weight_init :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     strides=stride, kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=he_init(),
                                     strides=stride, kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

            if norm_fn == 'instance' :
                x = instance_norm(x, 'res1_instance')
            if norm_fn == 'batch' :
                x = batch_norm(x, is_training, 'res1_batch')

            x = relu(x)
        with tf.variable_scope('res2') :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])

            if normal_weight_init :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     strides=stride, kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))
            else :
                x = tf.layers.conv2d(inputs=x, filters=channels, kernel_size=kernel, strides=stride,
                                     kernel_regularizer=tf_contrib.layers.l2_regularizer(scale=0.0001))

            if norm_fn == 'instance' :
                x = instance_norm(x, 'res2_instance')
            if norm_fn == 'batch' :
                x = batch_norm(x, is_training, 'res2_batch')

        if dropout_ratio > 0.0 :
            x = tf.layers.dropout(x, rate=dropout_ratio, training=is_training)

        return x + x_init

def activation(x, activation_fn='leaky') :
    assert activation_fn in ['relu', 'leaky', 'tanh', 'sigmoid', 'swish', None]
    if activation_fn == 'leaky':
        x = lrelu(x)

    if activation_fn == 'relu':
        x = relu(x)

    if activation_fn == 'sigmoid':
        x = sigmoid(x)

    if activation_fn == 'tanh' :
        x = tanh(x)

    if activation_fn == 'swish' :
        x = swish(x)

    return x

def lrelu(x, alpha=0.01) :
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def relu(x) :
    return tf.nn.relu(x)

def sigmoid(x) :
    return tf.sigmoid(x)

def tanh(x) :
    return tf.tanh(x)

def swish(x) :
    return x * sigmoid(x)

def batch_norm(x, is_training=False, scope='batch_nom') :
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

def instance_norm(x, scope='instance') :
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def gaussian_noise_layer(mu):
    sigma = 1.0
    gaussian_random_vector = tf.random_normal(shape=tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
    return mu + sigma * gaussian_random_vector

def KL_divergence(mu) :
    # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, axis = -1)
    # loss = tf.reduce_mean(KL_divergence)
    mu_2 = tf.square(mu)
    loss = tf.reduce_mean(mu_2)

    return loss

def L1_loss(x, y) :
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss

def discriminator_loss(real, fake, smoothing=False, use_lasgan=False) :
    if use_lasgan :
        if smoothing :
            real_loss = tf.reduce_mean(tf.squared_difference(real, 0.9)) * 0.5
        else :
            real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0)) * 0.5

        fake_loss = tf.reduce_mean(tf.square(fake)) * 0.5
    else :
        if smoothing :
            real_labels = tf.fill(tf.shape(real), 0.9)
        else :
            real_labels = tf.ones_like(real)

        fake_labels = tf.zeros_like(fake)

        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_labels, logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    loss = real_loss + fake_loss

    return loss

def generator_loss(fake, smoothing=False, use_lsgan=False) :
    if use_lsgan :
        if smoothing :
            loss = tf.reduce_mean(tf.squared_difference(fake, 0.9)) * 0.5
        else :
            loss = tf.reduce_mean(tf.squared_difference(fake, 1.0)) * 0.5
    else :
        if smoothing :
            fake_labels = tf.fill(tf.shape(fake), 0.9)
        else :
            fake_labels = tf.ones_like(fake)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_labels, logits=fake))

    return loss

