import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras import layers
import time

import tensorflow as tf

import os
import json


BUFFER_SIZE = 60000
BATCH_SIZE = 256

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(60*60*100, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((60, 60, 100)))
    assert model.output_shape == (None, 60, 60, 100) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(50, (60, 60), padding='same', use_bias=False))
    print(model.output_shape)
    assert model.output_shape == (None, 60, 60, 50)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(25, (60, 60), strides=(2, 2), padding='same', use_bias=False))
    print(model.output_shape)
    assert model.output_shape == (None, 120, 120, 25)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (20, 20), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
    print(model.output_shape)
    assert model.output_shape == (None, 240, 240, 1)
    #model.add(layers.Sigmoid())

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(25, (20, 20), strides=(2, 2), padding='same',input_shape=(240,240,1)) )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(50, (20, 20), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(100, (20, 20), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.5))


    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    #model.add(layers.Sigmoid())

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim], mean=80,stddev=100)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as you go
        #display.clear_output(wait=True)
        generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        #if (epoch + 1) % 15 == 0:
        #checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                            'final',
                            seed)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    out = open('tfout/'+str(epoch)+'.json','w')
    reducer = lambda x: int(x * 127.5 + 127.5)
    data = np.apply_along_axis(reducer, 3, predictions.numpy())
    out.write(json.dumps(data.tolist()))


map_matrix = {
    "0":[],
    "1":[],
    "2":[],
    "3":[],
    "4":[],
    "5":[],
    "6":[],
    "7":[],
    "8":[],
    "9":[],
    "10":[],
    "11":[],
    "12":[],
    "13":[],
    "14":[],
    "15":[],
    "16":[],
    "17":[],
    "18":[],
    "19":[],
    "20":[],
    "21":[],
    "22":[],
    "23":[]
}
paths = list(os.scandir("./out"))

# Filling the matrix

for j in range(len(paths)):
    if paths[j].is_file():
        file = open('out/'+paths[j].name, 'r')
        data = json.loads(file.readline())
        tileset = str(data['tileset'])
        map_matrix[tileset].append([])
        pos = len(map_matrix[tileset]) - 1
        for i in range(0,data['height']):
            map_matrix[tileset][pos].append([])
            for k in range(0,data['width']):
                map_matrix[tileset][pos][i].append([data['map_data'][i*data['width']+k]])
            

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_map = generator(noise, training=False)


discriminator = make_discriminator_model()
decision = discriminator(generated_map)
print (decision)

# Porting to array 
seed = tf.random.normal([num_examples_to_generate, noise_dim], mean=80,stddev=100)
print(seed.shape)

#Bringing each sample to the dimensions of 240 x 240
for i in range(len(map_matrix['0'])):
    if len(map_matrix['0'][i]) < 240:
        while len(map_matrix['0'][i]) < 240:
            map_matrix['0'][i].append([[-1]] * 240)
    for k in range(len(map_matrix['0'][i])):
        if(len(map_matrix['0'][i][k]) < 240):
            while len(map_matrix['0'][i][k]) < 240:
                map_matrix['0'][i][k] += ([[-1]])
        elif len(map_matrix['0'][i][k]) > 240:
            map_matrix['0'][i][k] = map_matrix['0'][i][k][0:240]



print(len(map_matrix['0'][0][0]))

np_matrix_tile0 = np.asarray(map_matrix['0']).astype('float32')
np_matrix_tile0 = np_matrix_tile0.reshape((-1,544,240,240,1))
print(np_matrix_tile0.shape)
#print(np_matrix_tile0.tolist()[0][0])
train(np_matrix_tile0, EPOCHS)


"""
model = keras.Sequential()

# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()
"""