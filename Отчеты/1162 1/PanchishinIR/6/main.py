#Анализ настроений

import warnings
warnings.simplefilter('ignore')

import matplotlib
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from keras import models
from keras import layers

import numpy as np


from keras.datasets import imdb

#костыль
old = np.load
np.load = lambda *a,**k: old(*a, **k, allow_pickle=True)

#написанный отзыв отмечается отношением (положительное, отрицательное), для удобства
#слова в отзыве заменены индексом. Индекс тем выше, чем чаще это слово встречается в отзывах
#0 - неизвестное слово
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

print("Категории:", np.unique(targets))
print("Количество уникальных слов:", len(np.unique(np.hstack(data))))

lengths = [len(i) for i in data]
print("Средняя длина отзыва:", np.mean(lengths))
print("Стандартное отклонение:", np.std(lengths))

def vectorize(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #a[1, [0, 4, 9]] = 1 отметить единицами указанные позиции
        results[i, sequence] = 1
    return results

data = vectorize(data)
targets = np.array(targets).astype("float32")

#print(len(targets)) #50 000
test_x = data[40000:]
test_y = targets[40000:]

train_x = data[:40000]
train_y = targets[:40000]


from keras import models
from keras import layers

model = models.Sequential()
# Входной слой
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))

# Скрытые слои
#исключение и коэффициент исключения
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))

model.add(layers.Dropout(0.2))
model.add(layers.Dense(50, activation = "relu"))

# Выходной слой
model.add(layers.Dense(1, activation = "sigmoid"))

# Описание модели
model.summary()

#binary_crossentropy так как работаем с бинарной классификацией
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=2, batch_size=500)

test_loss, test_acc = model.evaluate(test_x, test_y)
print('test_acc:', test_acc)
print('test_loss:', test_loss)

print("Модель говорит:", np.round(model.predict(test_x[1:2])[0]))
print("А на самом деле:", test_y[1:2])
