import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.utils.vis_utils import plot_model


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def create_generator_unet(image_size, weight_init=None):
    # image_input
    image_input = layers.Input(shape=(image_size, image_size, 3))

    mask_input = layers.Input(shape=(image_size, image_size, 1))

    concatenated_input = layers.concatenate([image_input, mask_input])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 64, 64, 64)
        downsample(128, 4),  # (batch_size, 32, 32, 128)
        downsample(256, 4),  # (batch_size, 16, 16, 256)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
        # downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    output_channels = 3
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = concatenated_input

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    model = tf.keras.Model(inputs=[image_input, mask_input], outputs=x)
    # print(model.summary())
    return model


def create_discriminator(image_size, weight_init):
    # define inputs
    image_input = layers.Input(shape=(image_size, image_size, 3))

    x = layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    # x = Dropout(0.25)(x)

    x = layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    # x = Dropout(0.25)(x)

    x = layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=weight_init)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    # x = Dropout(0.25)(x)

    # flatten input into 1-D and output a single a number from the last layer using sigmoid activation
    x = layers.Flatten()(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    # create the model
    model = tf.keras.Model(inputs=image_input, outputs=x)

    plot_model(model, to_file='model_discriminator.png', show_shapes=True, show_layer_names=True)

    # print(model.summary())
    return model
