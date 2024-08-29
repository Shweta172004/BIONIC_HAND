import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
import tflite_runtime
from tflite_runtime.interpreter import Interpreter
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout, Flatten, Dense, LSTM
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import os


# Path to the directory containing CSV files on your Google Drive
csv_dir = "path of the dataset "

# Read and concatenate all CSV files
all_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
combined_df = pd.concat([pd.read_csv(f, header=None) for f in all_files], ignore_index=True)

# Assign columns names
combined_df.columns = [f'col_{i+1}' if i != 1000 and i != 1001 else 'label' if i == 1000 else 'EMG' for i in range(combined_df.shape[1])]
# Separate features and labels
dataset_features_ch1 = combined_df[combined_df['EMG'] == 'EMG1']
dataset_labels = dataset_features_ch1['label']
dataset_features = dataset_features_ch1.drop(columns=['label','EMG'],axis=1)

# Encode labels
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(dataset_labels)
y = to_categorical(encoded_Y)

x = np.array(dataset_features[:])
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(dataset_labels)
encoded_Y = encoder.transform(dataset_labels)

# Convert integers to dummy variables (i.e. one hot encoded)
y = to_categorical(encoded_Y)
label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

kfold = KFold(n_splits=5, shuffle=True)
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x.shape[1],1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#cross validation
#for i, (train_index, test_index) in enumerate(kf.split(x)):
Results = {'accuracy':[], 'loss':[], 'val_accuracy':[], 'val_loss':[]}
#Results_Acc = []
fold_no = 1
for train, test in kfold.split(x, y):
    print('------------------------------------------------------------------------')
    print(f"Training for fold {fold_no} ...")
    #history = model.fit(x_r[train], y_r[train], epochs=100, batch_size=100, validation_data= (x_r[test], y_r[test]) )
    #history = model.fit(train, epochs=100, batch_size=100, validation_data= test )
    history = model.fit(x[train], y[train], epochs=100, batch_size=100, verbose=1, validation_data= (x[test], y[test]) )
    #scores = model.evaluate(x[test], y[test], verbose=0)

    # Increase fold number
    fold_no = fold_no + 1
    #Results = appendHist(Results, history)
    #Results_Acc = Results_Acc.append(history.history['accuracy'])
    Results['accuracy'].append(history.history['accuracy'])
    Results['loss'].append(history.history['loss'])
    Results['val_accuracy'].append(history.history['val_accuracy'])
    Results['val_loss'].append(history.history['val_loss'])

A = Results['accuracy']
B = Results['val_accuracy']
C = Results['loss']
D = Results['val_loss']

#Ac = np.array([A[0], A[1], A[2]])
TrainAcc = np.concatenate((A[0], A[1],A[2]), axis=0)
TestAcc = np.concatenate((B[0], B[1],B[2]), axis=0)
Trainloss = np.concatenate((C[0], C[1],C[2]), axis=0)
Testloss = np.concatenate((D[0], D[1],D[2]), axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2)
#fig.suptitle('Model Accuracy and Loss')
fig.set_figheight(4)
fig.set_figwidth(12)
ax1.plot(TrainAcc,'b')
ax1.plot(TestAcc,'r-')
ax1.set_ylim(-0.2, 1.2)
ax1.set_title('Model Accuracy')
ax1.set(xlabel='Epochs',ylabel='Accuracy')
ax1.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
ax2.plot(Trainloss,'b')
ax2.plot(Testloss,'r-')
#ax2.set_ylim(-0.2, 1.2)
ax2.set_title('Model Loss')
ax2.set(xlabel='Epochs',ylabel='Loss')
#plt.ylabel('Accuracy')

ax2.legend(['Train Loss', 'Test Loss'], loc='upper right')
#plt.show()

TF_LITE_MODEL_FILE_NAME = "path_to_save_the_model.tflite"
tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tf_lite_converter.convert()
tflite_model_name = TF_LITE_MODEL_FILE_NAME
open(tflite_model_name, "wb").write(tflite_model)


intrep = Interpreter(model_path="path_to_model.tflite")
intrep.allocate_tensors()

TF_LITE_MODEL_FLOAT_16_FILE_NAME = "path_to_16_bit_model.tflite"

tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)

tf_lite_converter.optimizations = [tf.lite.Optimize.DEFAULT]

tf_lite_converter.target_spec.supported_types = [tf.float16]


tflite_model = tf_lite_converter.convert ()

TF_LITE_SIZE_QUANT_MODEL_FILE_NAME = "path_to_save_the_quantised_model.tflite"

tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_model = tf_lite_converter.convert()

tflite_model_name = TF_LITE_SIZE_QUANT_MODEL_FILE_NAME
open(tflite_model_name, "wb").write(tflite_model)
