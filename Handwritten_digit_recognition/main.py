import os , time

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D  # new!

import cv2
import numpy as np
from skimage.transform import resize

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32')
X_train /= 255
X_test /= 255
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

filepath = './keras_model.h5'
if os.path.isfile(filepath):
    print("找到模型。")
    reload_model = tf.keras.models.load_model('./keras_model.h5')
else:
    print("沒找到模型，開始訓練。")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test, y_test))
    model.save('./keras_model.h5')
    time.sleep(1)
    reload_model = tf.keras.models.load_model('./keras_model.h5')

drawing = False  # 是否開始畫圖


# 滑鼠的回撥函式的引數格式是固定的，不要隨意更改。
def mouse_event(event, x, y, flags, param):
    global drawing

    # 左鍵按下：開始畫圖
    if event == 1:
        drawing = True
        start = (x, y)
    # 滑鼠移動，畫圖
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if y < 660 - 22:
                cv2.rectangle(img_main, (x, y), (x + 24, y + 24), (0, 0, 0), -1)

    elif event == 4:
        drawing = False


img_main = np.zeros((800, 560, 3), np.uint8)
img_draw = img_main[100:660, 0:560]
cv2.rectangle(img_main, (0, 0), (560, 100), (125, 125, 125), -1)
cv2.rectangle(img_main, (0, 100), (560, 660), (255, 255, 255), -1)
cv2.rectangle(img_main, (0, 660), (560, 800), (125, 125, 125), -1)

title_text = 'Handwritten digit recognition'
cv2.putText(img_main, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

while True:
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_event)
    cv2.imshow('image', img_main)
    image = cv2.resize(img_draw, (28, 28), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰階
    cv2.imshow('image2', image)
    image_resized = resize(image, (28, 28), anti_aliasing=True)
    X1 = image_resized.reshape(1, 28, 28)  # / 255
    X1 = np.abs(1 - X1)
    predictions = reload_model.predict(X1)
    pre_num = np.argmax(predictions[0])
    pre_pro = predictions[0, pre_num]
    pre_pro = round(pre_pro, 4)
    print(pre_num, pre_pro)

    if cv2.waitKey(1) == ord('r'):
        cv2.rectangle(img_main, (0, 100), (560, 660), (255, 255, 255), -1)

    if cv2.waitKey(1) == 27:
        break
