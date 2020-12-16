import tensorflow as tf

from dOD.tf_model import layers


class TestSequentialConv2DLayer:

    def test_serialization(self):
        seq_conv2d = layers.SequentialConv2DLayer(
            (128, 128, 64),
            nlayer=2,
            padding='same',
            strides=1,
            activation='relu',
            drop_rate=0.5,
            norm_type='instancenorm',
            name='seq_conv2d_test'
        )

        config = seq_conv2d.get_config()
        reconstructed = layers.SequentialConv2DLayer.from_config(config)

        assert reconstructed.kernel_shape == seq_conv2d.kernel_shape
        assert reconstructed.nlayer == seq_conv2d.nlayer
        assert reconstructed.padding == seq_conv2d.padding
        assert reconstructed.strides == seq_conv2d.strides
        assert reconstructed.activation == seq_conv2d.activation
        assert reconstructed.drop_rate == seq_conv2d.drop_rate
        assert reconstructed.norm_type == seq_conv2d.norm_type

        assert len(reconstructed.layers) == len(seq_conv2d.layers)


class TestConv2DTransposeLayer:

    def test_serialization(self):
        conv2d_transpose = layers.Conv2DTransposeLayer(
            (2, 2, 64),
            padding='same',
            strides=2,
            activation='relu',
            drop_rate=0.2,
            norm_type='batchnorm',
            name='conv2d_transpose_test'
        )

        config = conv2d_transpose.get_config()
        reconstructed = layers.Conv2DTransposeLayer.from_config(config)

        assert reconstructed.kernel_shape == conv2d_transpose.kernel_shape
        assert reconstructed.padding == conv2d_transpose.padding
        assert reconstructed.strides == conv2d_transpose.strides
        assert reconstructed.activation == conv2d_transpose.activation
        assert reconstructed.drop_rate == conv2d_transpose.drop_rate
        assert reconstructed.norm_type == conv2d_transpose.norm_type

        assert len(reconstructed.layers) == len(conv2d_transpose.layers)


class TestCropConcatLayer:

    def test_call(self):
        crop_concat = layers.CropConcatLayer()
        x1 = tf.random.uniform(shape=[42, 54, 54, 30])
        x2 = tf.random.uniform(shape=[42, 48, 48, 32])
        res = crop_concat(x1, x2)

        assert res.shape == (
            x2.shape[0],
            x2.shape[1],
            x2.shape[2],
            x1.shape[3] + x2.shape[3]
        )


class TestCropToShape:
    def test_crop_to_shape(self):
        X = tf.random.uniform(shape=[54, 54])
        shape = (42, 42, 3)
        res = layers.crop_to_shape(X, shape)

        assert res.shape == (shape[0], shape[1])
