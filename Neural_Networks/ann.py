import numpy as np
from utilities.cluster_creator import ClusterCreator
from utilities.data_sanitizer import data_import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


cluster_creator = ClusterCreator.build_clusters()
k_clusters = cluster_creator.k_cluster_dict
func_clusters = cluster_creator.func_cluster_dict
biome_clusters = cluster_creator.biome_cluster_dict

my_features = ['ta', 'rh', 'vpd', 'ppfd_in', 'swc_shallow', 'precip']
my_files = func_clusters[0]  # can select using the cluster dictionaries or use [] for all

# Import data and scale X inputs
X, Y = data_import(my_features, my_files)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#  Split to training/validation sets: 80% training, 10% test, 10% validation
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

# Set random seeds for reproducibility
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Build neural network
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', name='hidden1', input_shape=X_train.shape[1:]),
    keras.layers.Dense(20, activation='relu', name='hidden2'),
    keras.layers.Dense(20, activation='relu', name='hidden3'),
    keras.layers.Dense(20, activation='relu', name='hidden4'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, name='output', activation=None)
])

model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr=0.0001), metrics=['mae'])
early_stopping_cb = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
history = model.fit(X_train, Y_train, epochs=30, batch_size=20, validation_data=(X_val, Y_val), callbacks=early_stopping_cb)

mse_test, mae_test = model.evaluate(X_test, Y_test)
Y_pred = model.predict(X_test)
r2 = r2_score(Y_test, Y_pred)
Y_pred_train = model.predict(X_train)
r2_train = r2_score(Y_train, Y_pred_train)
print(f'R2 was {r2}')
print(f'MSE was {mse_test}')
print(f'MAE was {mae_test}')
print(f'R2 of the training set was {r2_train}')
if r2_train > r2:
    print(f'Model may be overfitting because {r2_train} > {r2}')

plt.scatter(Y_test, Y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
