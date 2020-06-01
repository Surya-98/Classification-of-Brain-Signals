import numpy as np
import pickle
import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from keras import metrics
import tensorflow as tf
from tensorflow.keras import layers
from IPython.display import SVG
from tensorflow.keras.utils import model_to_dot
from tensorflow.keras.utils import plot_model
# from tensorflow.python.util import nest
from keras.utils import to_categorical
from keras import optimizers
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from keras.callbacks import ModelCheckpoint

window_size = 10
overlap = 5
batch_size = 1
n_classes = 5
def preprocess(s):
    # print(len(s[0]))
    signal_2d = []
    for i in range(window_size):
        s[i] = stats.zscore(s[i])
        temp = np.array([[0, 0, 0, 0, s[i][21], s[i][22], s[i][23], 0, 0, 0, 0],
                    [0, 0, 0, s[i][24], s[i][25], s[i][26], s[i][27], s[i][28], 0, 0, 0],
                    [0, s[i][29], s[i][30], s[i][31], s[i][32], s[i][33],
                    s[i][34], s[i][35], s[i][36], s[i][37], 0],
                    [0, s[i][38], s[i][0], s[i][1], s[i][2], s[i][3], s[i][4],
                    s[i][5], s[i][6], s[i][39], 0],
                    [s[i][42], s[i][40], s[i][7], s[i][8], s[i][9], s[i][10],
                    s[i][11], s[i][12], s[i][13], s[i][41], s[i][43]],
                    [0, s[i][44], s[i][14], s[i][15], s[i][16], s[i][17], s[i][18],
                    s[i][19], s[i][20], s[i][45], 0],
                    [0, s[i][46], s[i][47], s[i][48], s[i][49], s[i][50],
                    s[i][51], s[i][52], s[i][53], s[i][54], 0],
                    [0, 0, 0, s[i][55], s[i][56], s[i][57], s[i][58], s[i][59], 0, 0, 0],
                    [0, 0, 0, 0, s[i][60], s[i][61], s[i][62], 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, s[i][63], 0, 0, 0, 0, 0]])
        signal_2d.append(temp)
    signal_2d = np.asarray(signal_2d)
    signal_2d = np.expand_dims(signal_2d, axis=3)
    return signal_2d

model = tf.keras.Sequential()
model.add(layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu', padding='same'), batch_input_shape=(batch_size, window_size,10,11,1)))
model.add(layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu', padding='same')))
model.add(layers.TimeDistributed(layers.Conv2D(128, (3,3), activation='relu', padding='same')))
model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
model.add(layers.TimeDistributed(layers.Flatten()))
model.add(layers.TimeDistributed(layers.Dense(1024)))
model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
model.add(layers.LSTM(64, activation='tanh', unroll=True,stateful=False,return_sequences=True))
model.add(layers.LSTM(64, activation='tanh', unroll=True,stateful=False))
model.add(layers.Dropout(0.5, noise_shape=None, seed=None))
model.add(layers.Dense(5, activation='softmax'))
model.summary()


# data_x goes from 0 to 108
checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1)

model.compile(loss='categorical_crossentropy',
             optimizer='adamax', metrics=[metrics.categorical_accuracy])


model.load_weights('model.weights.best.hdf5')

# 0th dimension - 2 (data, label)
# 1st dimension - 14 (experiment)
# 2nd dimension - (number of labels)
# 3rd dimension - (number of values in each label)
# 4th dimension - 64 (channels)

for i0 in range(109):
    file_name = './dataset/data_txt/data_'+str(i0)+'.txt'
    with open(file_name, 'rb') as F:
        L = pickle.load(F)
    F.close()
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i1 in range(14):
        print("Subject number:", i0)
        print("Experiment number:", i1)
        
        for i2 in range(len(L[0][i1])):
            for i3 in range(0, len(L[0][i1][i2])-window_size, overlap):
                if (i1<10 and i1!=1):
                    p = 0
                else:
                    x_test.append(preprocess(L[0][i1][i2][i3:i3+window_size]))
                    y_test.append(L[1][i1][i2])

    print("Before Training Validation:", len(x_test))
    val = len(x_test)
    for j in range(batch_size - val%batch_size):
        x_test.append(x_test[-1])
        y_test.append(y_test[-1])
        
y_test = to_categorical(y_test, num_classes=n_classes)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
y_pred = model.predict(x_test)
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)
print(y_pred)
print(y_test)
print(classification_report(y_test, y_pred))
print("Over")