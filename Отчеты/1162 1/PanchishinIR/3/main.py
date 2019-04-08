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

print('Простая линейная регрессия')
print('Коэффициенты', reg.coef_) # вывести коэффициенты
print('Коэф. детерм.', reg.score([[x, 1] for x in X[feature]], Y)) # вывести коэффициент детерминации

points = np.linspace(X[feature].min(), X[feature].max(), 100) # значения для предсказания
target = [np.sum(reg.coef_*[x, 1]) for x in points] # предсказания модели

plt.figure(figsize=(20, 10))
plt.scatter(X[feature], Y)
plt.plot(points, target, c='r')

### ваш код

feature = 2

def get_sample(x):
    return [x**4, x**3, x**2, x, 1]

reg = linear_model.LinearRegression(fit_intercept=False)
reg.fit([get_sample(x) for x in X[feature]], Y)

print('\nНелинейная простая регрессия')
print('Коэффициенты', reg.coef_)
print('Коэф. детерм.', reg.score([get_sample(x) for x in X[feature]], Y))

points = np.linspace(X[feature].min(), X[feature].max(), 100)
target = [np.sum(reg.coef_*get_sample(x)) for x in points]

plt.figure(figsize=(20, 10))
plt.scatter(X[feature], Y)
plt.plot(points, target, c='r')

###

def get_RX(x1, x2):
    return [[ x1[i]**2, x1[i], x2[i]**2, x2[i], 1 ] for i,_ in enumerate(x1)]

reg = [None, None, None]
for i,j in [(i1,j1) for i1 in range(10) for j1 in range(10)]:
    lreg = linear_model.LinearRegression(fit_intercept=False)
    RX = get_RX(X[i], X[j])
    lreg.fit(RX, Y)
    lscore = lreg.score(RX, Y)

    if reg[0] == None or reg[1] < lscore:
        reg = [lreg, lscore, (i, j)]

print('\nНелинейная множественная регрессия')
print('Коэффициенты', reg[0].coef_)
print('Коэф. детерм.', reg[1])
i,j = reg[2]
reg = reg[0]

x1 = np.linspace(X[i].min(), X[i].max(), 100)
x2 = np.linspace(X[j].min(), X[j].max(), 100)
x1,x2 = np.meshgrid(x1, x2)
RX = get_RX(x1,x2)
y = np.asarray([ np.sum(reg.coef_*x) for x in RX ])

from mpl_toolkits.mplot3d import Axes3D

plt.figure()
ax = Axes3D(plt.gcf())
ax.scatter(X[i], X[j], Y, c='r')
ax.plot_surface(x1, x2, y)

from sklearn.metrics import mean_absolute_error
y_true = Y
y_pred = [ np.sum(reg.coef_*x) for x in get_RX(X[i], X[j]) ]
mae = mean_absolute_error(y_true, y_pred)
print(mae)

###

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
    X.T, Y, test_size=0.2, random_state=0)

from sklearn import svm
model = svm.SVR()
preds = model.fit(train_X, train_y).predict(test_X)
mae = mean_absolute_error(test_y, preds)
print('\nОпорный вектор')
print('Ср. ар. предсказаний', np.mean(preds))
print('Ошибка', mae)

from sklearn import tree
model = tree.DecisionTreeRegressor(random_state=0)
preds = model.fit(train_X, train_y).predict(test_X)
mae = mean_absolute_error(test_y, preds)
print('\nЛес')
print('Ср. ар. предсказаний', np.mean(preds))
print('Ошибка', mae)

###

from sklearn.model_selection import cross_val_score
model = tree.DecisionTreeRegressor(random_state=0)
scores = cross_val_score(model, train_X, train_y, cv=5) # здесь подгонка не осуществляется, модель внутри клонируется
# тут нужно параметры модели подбирать (которые определяются НЕ в процессе обучения - например, степень полинома против его коэффициентов)
# проводится на тренировочных данных, так как подбирать параметры на всех данных - значит подгонять модель под данные (overfitting)
# в таком случае использование тестовой выборки для оценки точности будет некорректным - модель уже знакома с этими данными
# модель считается хорошей, если она хорошо прогнозирует данные
print('\nТочность на фрагментах', scores)
preds = model.fit(train_X, train_y).predict(test_X)
mae = mean_absolute_error(test_y, preds)
#acc = np.mean(test_y == preds) # не классификатор, очень вряд ли будет точное попадание
#print('Ошибка', mae)



plt.show()
