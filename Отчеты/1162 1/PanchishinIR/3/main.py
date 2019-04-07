import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

data = datasets.load_diabetes() # загрузим данные
Y, X = data['target'], data['data'].T
features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'] # признаки в датасете

### ваш код

for i, val in enumerate(features):
    plt.subplot(2, 5, i+1)
    plt.scatter(x=X[i], y=Y, s=10)
    plt.xlabel(val)
    plt.ylabel('target')

###

from sklearn import linear_model

feature = 2

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit([[x, 1] for x in X[feature]], Y) # обучим регрессию k*x + b*1, метод осуществит подбор коэффциентов k и b
print(reg.coef_) # вывести коэффициенты
print(reg.score([[x, 1] for x in X[feature]], Y)) # вывести коэффициент детерминации

points = np.linspace(X[feature].min(), X[feature].max(), 100) #значения для предсказания
target = [np.sum(reg.coef_*[x, 1]) for x in points] #предсказания модели

plt.figure(figsize=(20, 10))
plt.scatter(X[feature], Y)
plt.plot(points, target, c='r')

### ваш код
# нелинейная простая регрессия

feature = 2

def get_sample(x):
    return [x**4, x**3, x**2, x, 1]

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit([get_sample(x) for x in X[feature]], Y)
print(reg.coef_)
print(reg.score([get_sample(x) for x in X[feature]], Y))

points = np.linspace(X[feature].min(), X[feature].max(), 100)
target = [np.sum(reg.coef_*get_sample(x)) for x in points]

plt.figure(figsize=(20, 10))
plt.scatter(X[feature], Y)
plt.plot(points, target, c='r')

# нелинейная множественная регрессия

reg = linear_model.LinearRegression(fit_intercept=False)

#RX = []
#for i in X:
#    RR = []
#    for j in [0, 1]:
#        RR.extend([ i[j]**2, i[j] ])
#    RX.append(RR)

print([[1,2,3],[4,5,6]])

RX = (X[0]*X[0]).concatenate(X[1], axe=1)
print(RX)
#reg.fit(RX, Y)
#
#print(reg.coef_)
#print(reg.score(RX, Y))
#
#RX1 = np.linspace(X[0].min(), X[0].max(), 100)
#RX2 = np.linspace(X[1].min(), X[1].max(), 100)
#X1,X2 = meshgrid(RX1, RX2)
#target = [np.sum(reg.coef_*get_sample(x)) for x in points]
#
#plt.figure(figsize=(20, 10))
#plt.scatter(X[feature], Y)
#plt.plot(points, target, c='r')
#
#plt.show()
