
import tensorflow as tf

# the shape of the tensor is [batch, width, height, channels]

def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1,n,n,1], strides=[1,n,n,1],padding='SAME')

def conv2d(x, kernal, bias, keep_prob):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, kernal, strides=[1,1,1,1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, bias)
        return tf.nn.dropout(conv_2d_b, keep_prob)
       
def deconv2d(x, kernal, stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2]
        #output_shape = tf.stack(x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2)
        return tf.nn.conv2d_transpose(x, kernal, output_shape, strides=[1, stride, stride, 1], padding="SAME", name ="conv2d_transpose")

def weight_variable(shape, stddev = 0.1, name = "weight"):
    initial = tf.random.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def bias_variable_const(shape, name="bias"):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name=name)

def bias_variable_randomNorm(shape, stddev=1.0, name="bias"):
    initial = tf.random.normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def cross_entropy( result, mask):
    return -tf.math.reduce_mean(mask*tf.log(tf.clip_by_value(result, 0.0, 1.0)),name="cross_entropy")

def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        offsets = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize
