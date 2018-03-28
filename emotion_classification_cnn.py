import scipy.io as scio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# load train_data
file = '/Users/xuepenghui/PycharmProjects/practice/abc.mat'
data = scio.loadmat(file)
sample = data['persons']
train_data = []
for i in range(101):
    temp = []
    for j in range(380):
        temp.append(sample[0][i][j])
    train_data.append(temp)
train_tag = []
for k in range(47):
    train_tag.append(0)
for m in range(54):
    train_tag.append(1)

train_data1 = np.zeros((101, 380, 75, 1))
for a in range(101):
    for b in range(380):
        for c in range(75):
            train_data1[a][b][c][0] = train_data[a][b][c]
train_tag1 = keras.utils.to_categorical(train_tag)

'''
# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
'''
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(380, 75, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mae', 'acc'])

model.fit(train_data1, np.array(train_tag1), batch_size=32, epochs=10, validation_split=0.1)
# score = model.evaluate(x_test, y_test, batch_size=32)
