import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from scipy.spatial.transform import rotation
np.set_printoptions(precision=3)

expected_data_size = 3
y_size = 1
x_size = 6 + 6 + 1 + 3
au = 149597870700

# direct to folder
folder_path = r"data_lambert_cut_2/"

# create data arrays
y = np.zeros(np.size(os.listdir(folder_path)), dtype = object)
x = np.copy(y)

# load in data
for i,file in enumerate(tqdm(os.listdir(folder_path))):

    contents = np.genfromtxt(folder_path + file, delimiter=",")
    x_array = np.zeros(x_size)

    values = contents[1:, -3:].reshape(1, -1, order='F')[0]
    pm = values[6:9]
    pe = values[:3]
    phase_angle = np.arccos(np.dot(pm,pe)/(np.linalg.norm(pm)*np.linalg.norm(pe)))

    # skip faulty transfers
    if np.size(contents)//7 >= expected_data_size and phase_angle > np.pi/2: # contents[2,3] <= 0.5:

        # y[i] = contents[1:,1][3:].reshape(1, -1, order = 'F')[0]/10**3

        values = contents[1:,-3:].reshape(1, -1, order = 'F')[0]
        x_array[:2] = values[:2] /au
        x_array[2] = values[2] /au
        x_array[3:5] = values[6:8] /au
        x_array[5] = values[8] /au
        x_array[6] = (values[13] - values[12]) /(365*24*3600)
        x_array[7:10] = values[3:6]/10**3
        x_array[10:13] = values[9:12]/10**3
        x_array[13:] = np.cross(values[:3], values[6:9]) / np.linalg.norm(np.cross(values[:3],values[6:9]))

        v_a = contents[1:, 2][3:].reshape(1, -1, order='F')[0]
        v_d = contents[1:, 1][3:].reshape(1, -1, order='F')[0]
        v_m = values[9:12]
        v_e = values[3:6]

        y[i] = [np.linalg.norm(v_e - v_d) + np.linalg.norm(v_m - v_a)]
        x[i] = x_array

# remove faulty data points
keep = [np.all(transfer != 0) for transfer in y]
y = y[keep]
x = x[keep]

y = np.concatenate(y).reshape(-1,y_size)
x = np.concatenate(x).reshape(-1,x_size)


# split train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1, test_size = 0.2)

# scaling
Scaler = QuantileTransformer()
Scaler = StandardScaler()
x_train_scaled = Scaler.fit_transform(x_train)
x_test_scaled = Scaler.transform(x_test)

# NN setup
NN = tf.keras.models.Sequential()
NN.add(tf.keras.layers.Dense(x_size, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal,
                                          bias_initializer=tf.keras.initializers.he_normal
                             ))
NN.add(tf.keras.layers.Dense(1000, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal,
                                          bias_initializer=tf.keras.initializers.he_normal
                             ))
NN.add(tf.keras.layers.Dense(1000, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal,
                                          bias_initializer=tf.keras.initializers.he_normal
                             ))
NN.add(tf.keras.layers.Dense(1000, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal,
                                          bias_initializer=tf.keras.initializers.he_normal
                             ))
NN.add(tf.keras.layers.Dense(y_size,  activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal,
                                          bias_initializer=tf.keras.initializers.he_normal
                             ))
NN.compile(optimizer="adam",
           loss="MeanSquaredError",
           metrics=["RootMeanSquaredError"])

NN.fit(x_train_scaled, y_train, epochs=4)
# NN.predict(x_train_scaled)[:10]
# y_train[:10]