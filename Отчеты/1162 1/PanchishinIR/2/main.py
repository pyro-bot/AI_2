import numpy as np

my_array = np.linspace(5, 100, 100/5)
print(my_array)

my_array = my_array[my_array%10==0]
print(my_array)

my_array = my_array.reshape((-1,2))
print(my_array)

my_snd_array = my_array - 5
print(my_snd_array)

result = np.concatenate((my_array, my_snd_array), 1) #ось
print(result)

#.T - транспонирование
#result[[1,2]]/=my_array[:4].T
result1 = result.copy()
result1[[1,2]]/=my_array[:4].T
print(result1)

#where вернет индексы трушных элементов
result = result[np.where(result1 == result)] 
print(result)

result = result.reshape((3, -1))
result[[0,2]] = result[[2,0]] #свап строк
print(result)

print(np.mean(result, axis=1))
print(np.median(result, axis=1))
print(np.std(result, axis=1))

# нормализация на максимальное значение вдоль строк
res = map(lambda line: line/max(line), result)
print(list(res))

for i in np.linspace(10, 1000000, 10):
    nall = 0
    nin = 0
    for _ in range(int(i)):
        point = np.random.rand(2)
        r = np.sqrt(point[0]**2 + point[1]**2)
        nall += 1
        if r <= 1:
            nin += 1
    print(nin/nall*4)

### pandas ###
import pandas as pd
data = pd.read_csv('../dataset/adult.txt', sep=', ', engine='python')

print(data.info())

print(data['sex'].value_counts().values)
#print(data[data.isna().any(axis=1)])

print(data.groupby('sex')['age'].mean().values)

print(data['native-country'].value_counts(normalize=True)['Cuba'])

#print(data.groupby(by=lambda row: row['sales'] > 
#print(data.groupby('salary')['age'].agg({'mean': np.mean, 'std': np.std}))
print(data.groupby('salary')['age'].agg([np.mean, np.std]))

hieduc = ['Bachelors', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate']
#print(data[data['salary'] == '>50K']['education'].unique())
print(data[data['salary'] == '>50K']['education'].isin(hieduc).value_counts(normalize=True))

print(data.groupby(['race', 'sex'])['age'].describe())

print(data[data['salary'] == '>50K']['marital-status'].str.startswith('Married').value_counts(normalize=True))

hours = data['hours-per-week'].max()
print(hours)
hard_workers = data[data['hours-per-week'] == hours]
print(len(hard_workers))
print(hard_workers['salary'].value_counts(normalize=True))

print( data.groupby(['native-country', 'salary'])['hours-per-week'].mean())
