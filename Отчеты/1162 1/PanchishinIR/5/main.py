import numpy as np
import pandas as pd
import catboost as cb
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(iris.head()) # четыре признака: sepal_length, sepal_width, petal_length, petal_width
print(pd.unique(iris.species)) # классы, которые нужно предсказать



##### TASK 1

### визуализируем
sns.pairplot(iris)

### препроцессим
print(f'Объектов с пропусками: {iris.isna().max(axis=1).sum()}') # ноль

print(f'Дубликатов: {iris.duplicated().sum()}')
iris = iris[~iris.duplicated()]
print('Удалил ради интереса')

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
le.fit(pd.unique(iris.species))

y = le.transform(iris['species'].values)
print('Закодированные ириски:', y)



##### TASK 2

# создаем новые признаки - бинарные
def iris_combine(x, y, xlabel, ylabel):
    global le, iris
    plt.figure()
    for cl, co in zip(le.classes_, ['ro', 'go', 'bo']):
        plt.plot(
                x[iris.species == cl], #x
                y[iris.species == cl], #y
                co, 
                label=cl)
    plt.legend()
    plt.xlabel(xlabel); plt.ylabel(ylabel)

iris_combine(iris.sepal_width, iris.sepal_length, 'sepal_width', 'sepal_length')
one = iris.apply(lambda row: 1 if (row.sepal_width > 3) & (row.sepal_length < 6) else 0, axis=1)

iris_combine(iris.petal_width, iris.petal_length, 'petal_width', 'petal_length')
two = iris.apply(lambda row: 1 if (row.petal_width < 1) & (row.petal_length < 2.5) else 0, axis=1)

iris_combine(iris.sepal_width, iris.petal_length, 'sepal_width', 'petal_length')
three = iris.apply(lambda row: 1 if row.petal_length < 2 else 0, axis=1)

#добавляем признаки 
iris = iris.assign(one=one, two=two, three=three)
print(iris.head())

#
X = iris.drop(labels=['species'], axis=1).values
print('Регрессоры', X[:5])



##### TASK 3

from catboost import CatBoostClassifier
model = CatBoostClassifier(
        iterations=6,
        # почему? Автор статьи, которую рекомендуется прочитать в методичке, советует использовать
        # функцию Гаусса, если нет требований к устойчивости. Это она.
        # Нет, это для регресси. Для классификации классической является логистическая функция.
        # Опять нет. Если классов несколько, то нужно использовать функцию MultiClass.
        loss_function='MultiClass') #'Logloss', #Quantile:alpha=2

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=443)

# указываем на бинарные данные
# на самом деле на категориальные признаки (яблоко - 4 (числовые))
cat_features = list(range(X.shape[1]-3, X.shape[1]))
print('cat_features', cat_features)

model.fit(
    X_train, y_train, cat_features=cat_features, 
)


# предсказываем
preds = model.predict(X_test)
print(preds) 
print(y_test)

from sklearn.metrics import accuracy_score
# стопрацентная!
print('Точность: {:.5f}'.format(
        accuracy_score(y_test, preds)))



plt.show()
