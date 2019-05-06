import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

# Загрузим наши ириски
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Заведём красивую трёхмерную картинку
fig = plt.figure(1, figsize=(6, 5)) #figsize - width, height in inches
plt.clf() # Clear the current figure
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134) #угол места (угол между горизонтом и верхней точкой предмета) и азимут (угол между направлением на север и к-н предмет)

plt.cla() # Clear the current axes

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X[y == label, 0].mean(), # позиция текста
              X[y == label, 1].mean() + 1.5,
              X[y == label, 2].mean(), 
              name, # текст
              horizontalalignment='center', # направление
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w')) # коробка, в которой текст будет, бэкграунд

# Поменяем порядок цветов меток, чтобы они соответствовали правильному
y_clr = np.choose(y, [1, 2, 0]).astype(np.float) # из второго массива выбираются элементы под индексами в y
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_clr, cmap=plt.cm.nipy_spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])





### описание данных. Модель справилось плохо, так как у нее не хватило сложности

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Выделим из наших данных валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, 
                                                    stratify=y, 
                                                    random_state=42)

# Для примера возьмём неглубокое дерево решений
clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)
print('Точность: {:.5f}'.format(accuracy_score(y_test, 
                                                preds.argmax(axis=1))))





from sklearn import decomposition
# Прогоним встроенный в sklearn PCA (метод главных компонент)
pca = decomposition.PCA(n_components=2)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered)
X_pca = pca.transform(X_centered)

# И нарисуем получившиеся точки в нашем новом пространстве
plt.figure()
plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0);




# Повторим то же самое разбиение на валидацию и тренировочную выборку (копипаст плохо!)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3,
                                                    stratify=y,
                                                    random_state=42)

clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)
print('Точность: {:.5f}'.format(accuracy_score(y_test,
                                                preds.argmax(axis=1))))

# сколько информации потеряли, выбрав всего две компоненты, которые объясняют только часть вариации, которую вызывают регрессоры
for i, component in enumerate(pca.components_):
    print("{} компонента: {}% от начальной дисперсии".format(
        i + 1,
        round(100 * pca.explained_variance_ratio_[i], 2)))
    print(" + ".join("%.3f x %s" % (value, name)
                     for value, name in zip(component,
                                            iris.feature_names)))



### кластеризация

# ка средних
#Выбрать количество кластеров ￼k￼, которое нам кажется оптимальным для наших данных.
#Высыпать случайным образом в пространство наших данных ￼k￼ точек (центроидов).
#Для каждой точки нашего набора данных посчитать, к какому центроиду она ближе.
#Переместить каждый центроид в центр выборки, которую мы отнесли к этому центроиду.
#Повторять последние два шага фиксированное число раз, либо до тех пор пока центроиды не "сойдутся" (обычно это значит, что их смещение относительно предыдущего положения не превышает какого-то заранее заданного небольшого значения).

# Начнём с того, что насыпем на плоскость три кластера точек
X = np.zeros((150, 2))

np.random.seed(seed=123)
X[:50, 0] = np.random.normal(loc=0.0, scale=.3, size=50) #loc - среднее, scale - стандартное отклонение
X[:50, 1] = np.random.normal(loc=0.0, scale=.3, size=50)

X[50:100, 0] = np.random.normal(loc=2.0, scale=.5, size=50)
X[50:100, 1] = np.random.normal(loc=-1.0, scale=.2, size=50)

X[100:150, 0] = np.random.normal(loc=-1.0, scale=.2, size=50)
X[100:150, 1] = np.random.normal(loc=2.0, scale=.5, size=50)

plt.figure(figsize=(5, 5))
plt.plot(X[:, 0], X[:, 1], 'bo');


# В scipy есть замечательная функция, которая считает расстояния (евклидово)
# между парами точек из двух массивов, подающихся ей на вход
from scipy.spatial.distance import cdist

# Прибьём рандомность и насыпем три случайные центроиды для начала
np.random.seed(seed=123)
centroids = np.random.normal(loc=0.0, scale=1., size=6)
centroids = centroids.reshape((3, 2))

cent_history = []
cent_history.append(centroids)

for i in range(3):
    # Считаем расстояния от наблюдений до центроид
    distances = cdist(X, centroids)
    # Смотрим, до какой центроиде каждой точке ближе всего
    labels = distances.argmin(axis=1)

    # Положим в каждую новую центроиду геометрический центр её точек
    centroids = centroids.copy()
    centroids[0, :] = np.mean(X[labels == 0, :], axis=0)
    centroids[1, :] = np.mean(X[labels == 1, :], axis=0)
    centroids[2, :] = np.mean(X[labels == 2, :], axis=0)

    cent_history.append(centroids)

# А теперь нарисуем всю эту красоту
plt.figure(figsize=(8, 8))
for i in range(4):
    distances = cdist(X, cent_history[i])
    labels = distances.argmin(axis=1)

    plt.subplot(2, 2, i + 1)
    plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'bo', label='Кластер #1')
    plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'co', label='Кластер #2')
    plt.plot(X[labels == 2, 0], X[labels == 2, 1], 'mo', label='Кластер #3')
    plt.plot(cent_history[i][:, 0], cent_history[i][:, 1], 'rX')
    plt.legend(loc=0)
    plt.title('Шаг {:}'.format(i + 1));

# подбираем количество кластеров
# чем меньше !квадрат расстояния! (или модуль) до центроида, тем лучше,
# но при минимизации количество кластеров становится равным количеству точек
# поэтому нужно смотреть на скорость уменьшения ошибки (расстояния до точек в кластере)
# если уменьшение будет происходить медленно, то нужно остановиться с увеличением кластеров

from sklearn.cluster import KMeans
inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, 8), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');


plt.show()
