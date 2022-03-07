import numpy as np
import os
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm
np.set_printoptions(precision=3)
import joblib
from sklearn.metrics import r2_score
import seaborn as sns
sns.set_theme()
sns.set_palette("rocket")

expected_data_size = 100
y_vel_size = 3
y_pos_size = 3
x_size = 6*6
random_set_size = 1000

# direct to folder
folder_path = r"data_constant_dt_solar_2/"

# create data arrays
y_vel = np.zeros(np.size(os.listdir(folder_path)[:500]) * random_set_size, dtype = object)
y_pos = np.copy(y_vel)
x = np.copy(y_vel)
au = 149597870700

counter = 0
# load in data
for i,file in enumerate(tqdm(os.listdir(folder_path)[:500])):
    contents = np.genfromtxt(folder_path + file, delimiter=",")
    contents[1:4, 1:] /= au
    contents[4:, 1:] /= 10**3

    # skip faulty transfers
    if np.size(contents)//7 >= expected_data_size:
        for j in range(random_set_size):
            rand_int = np.random.randint(7, np.shape(contents)[1]-1)
            y_vel[counter] = contents[4:,rand_int].reshape(1, -1, order = 'F')[0]
            y_pos[counter] = contents[1:4,rand_int].reshape(1, -1, order = 'F')[0]
            x[counter] = contents[1:,rand_int-6:rand_int].reshape(1, -1, order = 'F')[0]
            counter += 1

y_vel = np.concatenate(y_vel).reshape(-1,y_vel_size)
y_pos = np.concatenate(y_pos).reshape(-1,y_pos_size)
x = np.concatenate(x).reshape(-1,x_size)

# split train and test set
x_train, x_test, y_vel_train, y_vel_test, y_pos_train, y_pos_test = train_test_split(x, y_vel, y_pos, random_state = 1, test_size = 0.2)


# scaling
Scaler = StandardScaler()
Scaler2 = StandardScaler()
x_train_scaled = Scaler.fit_transform(x_train)
x_test_scaled = Scaler.transform(x_test)


# NN setup
NN_vel = tf.keras.models.Sequential()
NN_vel.add(tf.keras.layers.Dense(x_size, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN_vel.add(tf.keras.layers.Dense(100, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN_vel.add(tf.keras.layers.Dense(100, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN_vel.add(tf.keras.layers.Dense(y_vel_size,  activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN_vel.compile(optimizer="adam",
           loss="MeanSquaredError",
           metrics=["RootMeanSquaredError"])

# NN setup
NN_pos = tf.keras.models.Sequential()
NN_pos.add(tf.keras.layers.Dense(x_size, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN_pos.add(tf.keras.layers.Dense(100, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN_pos.add(tf.keras.layers.Dense(100, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN_pos.add(tf.keras.layers.Dense(y_pos_size,  activation=lambda x: tf.nn.leaky_relu(x, alpha=0.01),
                                          kernel_initializer=tf.keras.initializers.he_normal(),
                                          bias_initializer=tf.keras.initializers.he_normal()))
NN_pos.compile(optimizer="adam",
           loss="MeanSquaredError",
           metrics=["RootMeanSquaredError"])

NN_vel.fit(x_train_scaled, y_vel_train, epochs=2)
NN_pos.fit(x_train_scaled, y_pos_train, epochs=2)

y_vel_test_predict = NN_vel.predict(x_test_scaled)
y_pos_test_predict =NN_pos.predict(x_test_scaled)

fig, (ax1, ax2) = plt.subplots(1,2)



rvel = []
rpos = []

for i in range(y_pos_size):
    rpos.append(r2_score(y_pos_test_predict[:,i], y_pos_test[:,i]))
for i in range(y_vel_size):
    rvel.append(r2_score(y_vel_test_predict[:,i], y_vel_test[:,i]))

avel, bvel = np.polyfit(np.linalg.norm(y_vel_test_predict, axis=1),
                        np.linalg.norm(y_vel_test, axis=1), 1)
apos, bpos = np.polyfit(np.linalg.norm(y_pos_test_predict, axis=1),
                        np.linalg.norm(y_pos_test, axis=1), 1)
rmagvel = r2_score(np.linalg.norm(y_vel_test_predict, axis=1),
                   np.linalg.norm(y_vel_test, axis=1))
rmagpos = r2_score(np.linalg.norm(y_pos_test_predict, axis=1),
                   np.linalg.norm(y_pos_test, axis=1))
ax1.scatter(np.linalg.norm(y_pos_test_predict, axis=1),
            np.linalg.norm(y_pos_test, axis=1))
ax1.set_ylabel("True position magnitude [AU]", fontsize=32)
ax1.set_xlabel("Predicted position magnitude [AU]", fontsize=32)
x_arraypos = np.linspace(np.min(np.linalg.norm(y_pos_test, axis=1)),
                         np.max(np.linalg.norm(y_pos_test, axis=1)), 1000)
ax1.plot(x_arraypos, apos*x_arraypos+bpos, c = "r", ls = "--", label=f"R2: {rmagpos}")
ax1.legend(fontsize=32)
ax1.set_xticks(ax1.get_xticks(),labelsize=22)
ax1.set_yticks(ax1.get_yticks(),labelsize=22)
ax2.scatter(np.linalg.norm(y_vel_test_predict, axis=1),
            np.linalg.norm(y_vel_test, axis=1))
ax2.set_ylabel("True velocity magnitude [km/s]", fontsize=32)
ax2.set_xlabel("Predicted velocity magnitude [km/s]", fontsize=32)
x_arrayvel = np.linspace(np.min(np.linalg.norm(y_vel_test, axis=1)),
                         np.max(np.linalg.norm(y_vel_test, axis=1)), 1000)
ax2.plot(x_arrayvel, apos*x_arrayvel+bvel, c = "r", ls = "--", label=f"R2: {rmagvel}")
ax2.legend(fontsize=32)
fig.set_tight_layout("tight")
ax2.set_xticks(ax2.get_xticks(),labelsize=22)
ax2.set_yticks(ax2.get_yticks(),labelsize=22)

# NN_vel.save("model_3/model_vel")
# NN_pos.save("model_3/model_pos")
# joblib.dump(Scaler, "model_3/scaler_x")
