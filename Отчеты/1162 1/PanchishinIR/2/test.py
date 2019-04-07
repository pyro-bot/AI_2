import numpy as np

# создание массива из списка
list_1 = [0,1,2,3,4]
arr_1d = np.array(list_1)

# прибавление и вычитание числа
def minus_plus(minus: bool):
    global list_1, arr_1d
    for i in range(len(list_1)):
        list_1[i] += -1 if minus else 1
    arr_1d += -1 if minus else 1
    print(list_1)
    print(arr_1d)
minus_plus(True)
minus_plus(False)

# инициализация массивов
zeros = np.zeros(10)
ones = np.ones(10)
arange = np.arange(10)
linspace = np.linspace(0,1,5)
print('zeros', zeros)
print('ones', ones)
print('arange', arange)
print('linspace', linspace)

# сложение массивов и т.д (элементы попарно складываются, 
#если размерность совпадать не будет, то не выполнится)
print('ones + arange', ones + arange)
print('ones * arange', ones * arange)
print('ones / arange', ones / arange)

# срез
print(np.arange(10)[5:9])

# две по пять
arange = np.arange(10).reshape(2,5)
print(arange)
print(arange[:, 3:]) #все строки от третьего столбца
# вывести элементы, где истина
print(arange[arange%2==1])

# решение задачек в main.py

import pandas as pd

# создаем
df = pd.DataFrame({
        'int_col' : [1,2,6,8,-1], 
        'float_col' : [0.1, 0.2,0.2,10.1,None], 
        'str_col' : ['a','b',None,'c','a']})
print(df)

# индексируемся
print(df.loc[:,['float_col','int_col']]) #строки, от, до

# загружаемся
df = pd.read_csv('../dataset/churn.csv')
print(df.head())

# майним информацию
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
# статистку по заданным типам
print(df.describe(include=['object', 'bool']))

# сортируемся
df = df.sort_values(by='total day charge', ascending=False)
print(df.head())

# индексируемся по индексам (15 строк, 3 столбца)
print(df.iloc[:15, :3])
# индексируемся по именам
print(df.loc[:15])

# сортируемся по нескольким признакам
print(df.sort_values(by=['account length', 'total day charge'],
        ascending=[True, False]).head())

# среднее столбцов, где длинна аккаунта 1
print(df[df['account length'] == 1].mean())
#тут поинтереснее
print(df[(df['account length'] == 1) & (df['international plan'] == 'no')]
        ['total intl minutes'].max())

# группируемся!
columns_to_show = ['total day minutes', 'total eve minutes', 'total night minutes']
print(df.groupby(['churn'])[columns_to_show].describe(percentiles=[]))

# задачки в main.py!
