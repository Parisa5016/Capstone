import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import matplotlib.pyplot as plt

# Set random seeds
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load and scale the data
df_scaled = pd.read_csv(r"C:\Users\alire\Desktop\Capstone Project\2.Data Transformation (Scaling)\scaled_preprocessed_online_shoppers_intention_not_MinMax.csv")

# Drop the original non-numeric columns
df_scaled = df_scaled.drop(columns=['Month', 'VisitorType'])

# Define the Autoencoder model
input_dim = df_scaled.shape[1]           # Number of features (input dimension)
encoding_dim = 2                         # We want to reduce it to 2 dimensions

# Input layer
input_layer = Input(shape=(input_dim,))

# Encoding layer
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)

# Decoding layer
decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# Define the Autoencoder model
autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the Autoencoder
autoencoder.fit(df_scaled, df_scaled, epochs=50, batch_size=256, shuffle=True, validation_split=0.2, verbose=1)

# Encoder model to extract reduced dimensions
encoder = Model(input_layer, encoded)

# Apply the Encoder to the scaled data
df_autoencoded = encoder.predict(df_scaled)

# Convert the NumPy array to a Pandas DataFrame
df_autoencoded = pd.DataFrame(df_autoencoded) # Convert to DataFrame
df_autoencoded.to_csv(r"C:\Users\alire\Desktop\Capstone Project\4.Dimensionality_Reduction_Techniques\autoencoder_transformed_online_shoppers_intention_1.csv", index=False)

# Visualize the 2D reduced data
plt.scatter(df_autoencoded[0], df_autoencoded[1], c='blue', marker='o', alpha=0.7) # Access columns by index for DataFrame
plt.title('Autoencoder visualization of customers')
plt.xlabel('AutoEncoder Component 1')
plt.ylabel('AutoEncoder Component 2')
plt.show()