# STEP 1: Imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.manifold import TSNE
import seaborn as sns

# STEP 2: Load and preprocess MNIST
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# STEP 3: Build autoencoder model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),  # Latent space
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(784, activation='sigmoid')  # Output layer
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# STEP 4: Train with EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, X_train, epochs=30, batch_size=64, validation_data=(X_test, X_test), callbacks=[early_stop])

# STEP 5: Evaluate model
loss = model.evaluate(X_test, X_test)
print(f"Test Loss: {loss:.4f}")

# STEP 6: Extract encoder
encoder = Sequential([
    model.layers[0],
    model.layers[1],
    model.layers[2]
])
encoded_imgs = encoder.predict(X_test)
print("Encoded shape:", encoded_imgs.shape)

# STEP 7: Visualize latent space with t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(encoded_imgs)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1])
plt.title("t-SNE Projection of Encoded Features")
plt.show()

# STEP 8: Compute reconstruction error
reconstructed = model.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstructed), axis=1)

plt.figure(figsize=(10, 4))
plt.hist(reconstruction_error, bins=50, color='skyblue')
plt.title("Reconstruction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# STEP 9: Visualize top anomalies
top_anomalies = np.argsort(reconstruction_error)[-10:]

plt.figure(figsize=(20, 4))
for i, idx in enumerate(top_anomalies):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Err: {reconstruction_error[idx]:.4f}")
    plt.axis('off')

    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[idx].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.tight_layout()
plt.show()

# STEP 10: Save encoder for deployment
encoder.save("mnist_encoder_model.h5")
