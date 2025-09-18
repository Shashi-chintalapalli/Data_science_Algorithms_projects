import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------
#  Generator
# ---------------------
def build_generator(noise_dim=100):
    model = Sequential([
        Input(shape=(noise_dim,)),
        Dense(256),
        LeakyReLU(0.2),
        Dense(512),
        LeakyReLU(0.2),
        Dense(1024),
        LeakyReLU(0.2),
        Dense(28 * 28, activation='tanh'),
        Reshape((28, 28, 1))
    ])
    return model

# ---------------------
#  Discriminator
# ---------------------
def build_discriminator():
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Flatten(),
        Dense(512),
        LeakyReLU(0.2),
        Dense(256),
        LeakyReLU(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# ---------------------
#  Image Saving
# ---------------------
def save_images(generator, noise_dim, epoch, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)
    noise = np.random.normal(0, 1, (25, noise_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(5, 5, figsize=(5, 5))
    cnt = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1

    plt.tight_layout()
    filename = os.path.join(output_dir, f"epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()

# ---------------------
#  Training
# ---------------------
def train_gan(epochs=10000, batch_size=64, noise_dim=100, sample_interval=500):
    # Load and preprocess MNIST
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 127.5 - 1.0  # Normalize to [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)

    # Build models
    generator = build_generator(noise_dim)
    discriminator = build_discriminator()

    # Compile discriminator
    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=Adam(0.0002, 0.5),
        metrics=['accuracy']
    )

    # Build and compile combined model
    discriminator.trainable = False
    gan_input = Input(shape=(noise_dim,))
    generated_image = generator(gan_input)
    validity = discriminator(generated_image)
    combined = Model(gan_input, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    # Training loop
    half_batch = batch_size // 2
    for epoch in range(epochs + 1):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], half_batch)
        real_imgs = x_train[idx]

        noise = np.random.normal(0, 1, (half_batch, noise_dim))
        fake_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        # Logging and saving
        if epoch % sample_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")
            save_images(generator, noise_dim, epoch)

# ---------------------
#  Run Training
# ---------------------
train_gan()
