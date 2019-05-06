# кластеризировать будем классы билетов, так как там три кучки должно получиться, вместо двух
# в случае с выживанием

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# загружаем данные и избавляемся от нечисловых данных и того, что нельзя классифицировать
data = pd.read_csv('../dataset/titanic_train.csv', index_col='PassengerId')\
        .drop(['Ticket', 'Cabin', 'Name'], axis=1)

#print(data.columns)
print(data.shape)
print(data.head())
# оч много, дропать и брать среднее нельзя
print(f'Объектов с пропусками: {data.isna().max(axis=1).sum()}') 
print('Заполняем предыдущим валидным наблюдением') # плотность пропусков небольшая
data = data.fillna(method='pad')

# удаление числовой зависимости у классовых занчений и обработка строковых данных
y = data['Pclass'].values
#print(y)
data_X = pd.get_dummies(data.drop('Pclass', axis=1), columns=['Sex', 'Embarked'])
print('Рабочие регрессоры:')
print(data_X.head())
X = data_X.values
print('Теперь в сыром виде:')
print(X[:5])

# теперь можно выделить пару главных компонент, чтобы работать с двумерной плоскостью
from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
#print(X.mean(axis=0)) # вектор по столбцам
X_centered = X - X.mean(axis=0) 
pca.fit(X_centered)
X_pca = pca.transform(X_centered)

resfig = plt.figure()
plt.subplot(1,2,1)
plt.title('Наблюдаемые классы')
for pclass, color in zip(np.unique(y), ['ro', 'go', 'bo']): # o в конце - маркер
    plt.plot(X_pca[y==pclass, 0], X_pca[y==pclass, 1], color, label=pclass, markersize=4)
plt.legend()

# теперь выделим кластеры
from sklearn.cluster import KMeans

# нарисуем изменение ошибки
plt.figure()
inertia = []
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(X_pca)
    inertia.append(np.sqrt(kmeans.inertia_))
plt.plot(range(1, 8), inertia, marker='s');
plt.xlabel('$k$')
plt.ylabel('$J(C_k)$');

# выделим кластеры
plt.figure(resfig.number)
plt.subplot(1, 2, 2)
plt.title('Выделенные классы')

kmeans = KMeans(n_clusters=3, random_state=1).fit(X_pca)
preds = kmeans.predict(X_pca)

for pclass, color in zip(np.unique(preds), ['ro', 'go', 'bo']): # o в конце - маркер
    plt.plot(X_pca[preds==pclass, 0], X_pca[preds==pclass, 1], color, label=pclass, markersize=4)
plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 'yX')
plt.legend()



plt.show()


### метод справился отвратительно, так как классы не оформлены. 
### С выживаниями было бы лучше
#from sklearn.metrics import accuracy_score
#print('Точность: {:.5f}'.format(
#    accuracy_score(y, preds.argmax(axis=1))))
