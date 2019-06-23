import numpy as np
import pandas as pd
import keras as k
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Train images shape:", train_images.shape)
print("Train labels len:", len(train_labels))
print("Test images shape:", test_images.shape)
print("Test labels len", len(test_labels))

from keras import models
from keras import layers

network = models.Sequential()
#принимает пиксели
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
#возвращает вероятности принадлежности метке
network.add(layers.Dense(10, activation='softmax'))

#функция потерь - оценка качества работы сети
#оптимизатор - механизм обновления сети
#метрика оценки качества (точность, доля правильно классифицированных изображений)
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#переведем в одно измерение и масштабируем данные (от 255 к 1)
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


#one hot encoding
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
print('test_loss:', test_loss)


#вывод объекта
from keras.datasets import mnist
(train_images_raw, train_labels_raw), (test_images_raw, test_labels_raw) = mnist.load_data()

# Получение значения по изображению
# Цифра, распознанная нейронной сетью
digit = network.predict(test_images[1:2])[0].argmax()

# Действительное изображение
digit_raw = test_images_raw[1]

print(digit)
# Вывод изображения
plt.imshow(digit_raw, cmap=plt.cm.binary)
plt.show()
