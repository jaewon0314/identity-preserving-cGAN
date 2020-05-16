import os
import os
import time
import scipy.misc
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras import backend as K
from keras.applications import InceptionResNetV2
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization, Reshape, concatenate, LeakyReLU, Lambda, \
    Activation, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing import image
from scipy.io import loadmat
from tqdm import tqdm

def build_encoder():
    """
    Encoder Network
    """
    input_layer = Input(shape=(64, 64, 3))

    # 1st Convolutional Block
    enc = Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(input_layer)
    # enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 2nd Convolutional Block
    enc = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 3rd Convolutional Block
    enc = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # 4th Convolutional Block
    enc = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Flatten layer
    enc = Flatten()(enc)

    # 1st Fully Connected Layer
    enc = Dense(4096)(enc)
    enc = BatchNormalization()(enc)
    enc = LeakyReLU(alpha=0.2)(enc)

    # Second Fully Connected Layer
    enc = Dense(100)(enc)

    # Create a model
    model = Model(inputs=[input_layer], outputs=[enc])
    return model


def build_generator():
    """
    Create a Generator Model with hyperparameters values defined as follows
    """
    latent_dims = 100
    num_classes = 6

    input_z_noise = Input(shape=(latent_dims,))
    input_label = Input(shape=(num_classes,))

    x = concatenate([input_z_noise, input_label])

    x = Dense(2048, input_dim=latent_dims + num_classes)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Dense(256 * 8 * 8)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.2)(x)

    x = Reshape((8, 8, 256))(x)
    
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=128, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=5, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=3, kernel_size=5, padding='same')(x)
    x = Activation('tanh')(x)

    model = Model(inputs=[input_z_noise, input_label], outputs=[x])
    return model


def expand_label_input(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, 32, 32, 1])
    return x


def build_discriminator():
    """
    Create a Discriminator Model with hyperparameters values defined as follows
    """
    input_shape = (64, 64, 3)
    label_shape = (6,)
    image_input = Input(shape=input_shape)
    label_input = Input(shape=label_shape)

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(image_input)
    x = LeakyReLU(alpha=0.2)(x)

    label_input1 = Lambda(expand_label_input)(label_input)
    x = concatenate([x, label_input1], axis=3)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[image_input, label_input], outputs=[x])
    return model


def build_fr_combined_network(encoder, generator, fr_model):
    input_image = Input(shape=(64, 64, 3))
    input_label = Input(shape=(6,))

    latent0 = encoder(input_image)

    gen_images = generator([latent0, input_label])

    fr_model.trainable = False

    resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor=2, width_factor=2,
                                                      data_format='channels_last'))(gen_images)
    embeddings = fr_model(resized_images)

    model = Model(inputs=[input_image, input_label], outputs=[embeddings])
    return model


def build_fr_model(input_shape):
    resent_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    image_input = resent_model.input
    x = resent_model.layers[-1].output
    out = Dense(128)(x)
    embedder_model = Model(inputs=[image_input], outputs=[out])

    input_layer = Input(shape=input_shape)

    x = embedder_model(input_layer)
    output = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

    model = Model(inputs=[input_layer], outputs=[output])
    return model


def build_image_resizer():
    input_layer = Input(shape=(64, 64, 3))

    resized_images = Lambda(lambda x: K.resize_images(x, height_factor=3, width_factor=3,
                                                      data_format='channels_last'))(input_layer)

    model = Model(inputs=[input_layer], outputs=[resized_images])
    return model


def load_data(data_dir, dataset='celeba'):

    f = open("list_attr_celeba.txt", "r").readlines()
    y_label = [[0 if i == '-1' else 1 for i in x.split()[1:]] for x in f[2:]]
    images=[os.path.join(data_dir, '{:0>6}.jpg'.format(i)) for i in range(1,202600)]

    # Return a list of all images and respective age
    return images, y_label

def center_crop(img, crop_h=108, crop_w=108, resize_h=64, resize_w=64):
    
    h, w = img.shape[:2]
    h_start = int(round((h - crop_h) / 2.))
    w_start = int(round((w - crop_w) / 2.))
    # resize image
    img_crop = scipy.misc.imresize(img[h_start:h_start+crop_h, w_start:w_start+crop_w], [resize_h, resize_w])
    return img_crop

def load_batch(image_paths):
    images = None
    for i, image_path in enumerate(image_paths):
        try:
            # Load image
            loaded_image = image.load_img(image_path)

            # Convert PIL image to numpy ndarray
            loaded_image = image.img_to_array(loaded_image)
            loaded_image = center_crop(loaded_image)

            # Add another dimension (Add batch dimension)
            loaded_image = np.expand_dims(loaded_image, axis=0)

            # Concatenate all images into one tensor
            if images is None:
                images = loaded_image
            else:
                images = np.concatenate([images, loaded_image], axis=0)
        except Exception as e:
            print("Error:", i, e)
    return images


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def write_log(callback, name, value, batch_no):
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    callback.writer.add_summary(summary, batch_no)
    callback.writer.flush()


def save_rgb_img(y_label,imgs, path):
    """
    Save an rgb image
    """
    fig = plt.figure()

    for i,img in enumerate(imgs):
        ax = fig.add_subplot(5, 5, i+1)
        ax.imshow((img+1.)/2.)
        ax.axis("off")
        ax.set_title(''.join(str(x)for x in y_label[i]),fontsize= 3)
    if not os.path.isdir("../results"):
        os.makedirs("../results")
    plt.savefig(path)
    plt.close()


def save_opt_img(real_imgs,gen_imgs,path):

    fig = plt.figure()

    for i in range(4):
        ax1 = fig.add_subplot(4,2,2*i)
        ax1.imshow(real_imgs[i])
        ax1.axis("off")
        ax2 = fig.add_subplot(4,2,2*i+1)
        ax2.imshow((gen_imgs[i]+1.)/2.)
        ax2.axis("off")

    plt.savefig(path)
    plt.close()

    
if __name__ == '__main__':
    # Define hyperparameters
    data_dir = "../img_align_celeba"
    epochs = 100
    batch_size = 128
    image_shape = (64, 64, 3)
    z_shape = 100
    TRAIN_GAN = False
    TRAIN_ENCODER = True
    TRAIN_GAN_WITH_FR = False
    fr_image_shape = (192, 192, 3)

    # Define optimizers
    dis_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    gen_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    adversarial_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    """
    Build and compile networks
    """
    # Build and compile the discriminator network
    discriminator = build_discriminator()
    discriminator.compile(loss=['binary_crossentropy'], optimizer=dis_optimizer)

    # Build and compile the generator network
    generator = build_generator()
    generator.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    # Build and compile the adversarial model
    discriminator.trainable = False
    input_z_noise = Input(shape=(100,))
    input_label = Input(shape=(6,))
    recons_images = generator([input_z_noise, input_label])
    valid = discriminator([recons_images, input_label])
    adversarial_model = Model(inputs=[input_z_noise, input_label], outputs=[valid])
    adversarial_model.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    """
    Load the dataset
    """
    images, y = load_data(data_dir=data_dir)
    y=np.array(y)
    print(y.shape)
    loaded_images = []
    # Implement label smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32)*0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32)+0.1

    """
    Train the generator and the discriminator network
    """
    if TRAIN_GAN:
        iter_time=0
        for epoch in range(epochs):
            print("Epoch:{}".format(epoch))

            number_of_batches = int(len(images) / batch_size)
            for index in tqdm(range(number_of_batches)):
                if iter_time % 500 == 0 and iter_time > 0:

                    inds = np.random.choice(202599, batch_size,replace=False)
                    y_batch = np.array([y[i] for i in inds])
                    z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                    gen_images = generator.predict_on_batch([z_noise, y_batch])

                    save_rgb_img(y_batch[:25],gen_images[:25], path="../results/img_{}.png".format(iter_time))
                if epoch==0:
                    images_batch = load_batch(images[index * batch_size:(index + 1) * batch_size])
                    loaded_images.extend(images_batch)
                    y_batch = y[index * batch_size:(index + 1) * batch_size]
                else:
                    pick = np.random.choice(loaded_images.shape[0], batch_size, replace=False)
                    images_batch = loaded_images[pick]
                    y_batch = y[pick]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                """
                Train the discriminator network
                """

                # Generate fake images
                initial_recon_images = generator.predict_on_batch([z_noise, y_batch])

                d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
                d_loss_fake = discriminator.train_on_batch([initial_recon_images, y_batch], fake_labels)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # print("d_loss:{}".format(d_loss))

                """
                Train the generator network
                """

                z_noise2 = np.random.normal(0, 1, size=(batch_size, z_shape))
                y_label = y[np.random.choice(202599, batch_size, replace=False)]

                g_loss = adversarial_model.train_on_batch([z_noise2, y_label], [1] * batch_size)

                # print("g_loss:{}".format(g_loss))
                iter_time+=1
                write_log(tensorboard, 'g_loss', g_loss, iter_time)
                write_log(tensorboard, 'd_loss', d_loss, iter_time)  
            if epoch == 0:
                loaded_images.extend(load_batch(images[number_of_batches*batch_size :])) 
                loaded_images=np.array(loaded_images)
            try:
                if not os.path.isdir("../gen"):
                    os.makedirs("../gen")
                    os.makedirs("../dis")
                if(epoch%5==0):
                    generator.save_weights("../gen/generator_{}.h5".format(epoch))
                    discriminator.save_weights("../dis/discriminator_{}.h5".format(epoch))
            except Exception as e:
                print("Error:", e)
            
        # Save networks

    """
    Train encoder
    """

    if TRAIN_ENCODER:
        # Build and compile encoder
        iter_time=0
        encoder = build_encoder()
        encoder.compile(loss=euclidean_distance_loss, optimizer='adam')
        tensorboard.set_model(encoder)
        # Load the generator network's weights
        try:
            generator.load_weights("generator_20.h5")
        except Exception as e:
            print("Error:", e)

        z_i = np.random.normal(0, 1, size=(5000, z_shape))
        y = np.random.randint(0, 2, 6*5000,dtype=np.int64).reshape(-1, 6)

        for epoch in range(epochs):
            print("Epoch:", epoch)

            number_of_batches = int(z_i.shape[0] / batch_size)
            for index in tqdm(range(number_of_batches)):
                # print("Batch:", index + 1)

                z_batch = z_i[index * batch_size:(index + 1) * batch_size]
                y_batch = y[index * batch_size:(index + 1) * batch_size]

                generated_images = generator.predict_on_batch([z_batch, y_batch])

                # Train the encoder model
                encoder_loss = encoder.train_on_batch(generated_images, z_batch)
                iter_time+=1
                write_log(tensorboard, "encoder_loss", encoder_loss, iter_time)


        # Save the encoder model
        encoder.save_weights("encoder.h5")

    """
    Optimize the encoder and the generator network
    """
    if TRAIN_GAN_WITH_FR:

        # Load the encoder network
        encoder = build_encoder()
        encoder.load_weights("encoder.h5")

        # Load the generator network
        generator.load_weights("generator_20.h5")

        image_resizer = build_image_resizer()
        image_resizer.compile(loss=['binary_crossentropy'], optimizer='adam')

        # Face recognition model
        fr_model = build_fr_model(input_shape=fr_image_shape)
        fr_model.compile(loss=['binary_crossentropy'], optimizer="adam")

        # Make the face recognition network as non-trainable
        fr_model.trainable = False

        # Input layers
        input_image = Input(shape=(64, 64, 3))
        input_label = Input(shape=(6,))

        # Use the encoder and the generator network
        latent0 = encoder(input_image)
        gen_images = generator([latent0, input_label])

        # Resize images to the desired shape
        resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor=3, width_factor=3,
                                                          data_format='channels_last'))(gen_images)
        embeddings = fr_model(resized_images)

        # Create a Keras model and specify the inputs and outputs for the network
        fr_adversarial_model = Model(inputs=[input_image, input_label], outputs=[embeddings])

        # Compile the model
        fr_adversarial_model.compile(loss=euclidean_distance_loss, optimizer=adversarial_optimizer)
        
        iter_time=0
        for epoch in range(epochs):
            print("Epoch:", epoch)

            reconstruction_losses = []

            number_of_batches = int(len(loaded_images) / batch_size)
            for index in tqdm(range(number_of_batches)):
                # print("Batch:", index + 1)
                if iter_time % 500 == 0:
                    pick=np.random.choice(np.shape(loaded_images)[0], batch_size, replace=False)
                    y_batch = y[pick]
                    real_images = np.array([loaded_images[i] for i in pick])
                    z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                    gen_images = generator.predict_on_batch([z_noise, y_batch])
                    save_opt_img(real_images[:4],gen_images[:4], path="../drive/My Drive/resultsult/5_16_1_image_opt/img_opt_{}.png".format(iter_time))
                
                if epoch==0:
                    images_batch = load_batch(images[index * batch_size:(index + 1) * batch_size])
                    loaded_images.extend(images_batch)
                    y_batch = y[index * batch_size:(index + 1) * batch_size]
                else:
                    pick = np.random.choice(loaded_images.shape[0], batch_size, replace=False)
                    images_batch = loaded_images[pick]
                    y_batch = y[pick]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                images_batch_resized = image_resizer.predict_on_batch(images_batch)

                real_embeddings = fr_model.predict_on_batch(images_batch_resized)

                reconstruction_loss = fr_adversarial_model.train_on_batch([images_batch, y_batch], real_embeddings)

                # print("Reconstruction loss:", reconstruction_loss)
                iter_time+=1
                write_log(tensorboard, "reconstruction_loss", reconstruction_loss, iter_time)

         # Save improved weights for both of the networks
        generator.save_weights("generator_optimized.h5")
        encoder.save_weights("encoder_optimized.h5")
