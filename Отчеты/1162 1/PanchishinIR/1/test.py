import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
S,C = np.sin(X),np.cos(X)

# Создать новую фигуру(окно, контекст) размером 8x6 точек, используя 100 точек на дюйм
plt.figure(figsize=(8,6), dpi=100)

# Add a subplot to the current figure
# The axes of the subplot
ax1 = plt.subplot(2, 2, 1) #221 nrow, ncol, index
#delete ax2 from the figure
#plt.delaxes(ax2)
#add ax2 to the figure again
#plt.subplot(ax2)

# Построение Косинуса синим цветом с непрерывной линией шириной 1 (пиксели)
plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="Косинус")

# Синус участка с использованием зеленого цвета с непрерывной линией шириной 1 (пикселей)
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")

plt.legend(loc='upper left', frameon=False)

# Установить лимит для X
plt.xlim(-4.0,4.0)

# Установить количество тиков для X
plt.xticks(np.linspace(-4,4,9,endpoint=True))

#Установить лимит для Y
plt.ylim(-1.0,1.0)

# Установить количество тиков для Y
plt.yticks(np.linspace(-1,1,5,endpoint=True))

# Сохранить рисунок с разрешением 72 dpi
# savefig("figure.png",dpi=72)

#############

import seaborn as sns

plt.subplot(2, 2, 2)

# Загружаем данные
tips = sns.load_dataset("tips") #с интернета берет, из своего репозитория
# Создаем необходимый формат
sns.violinplot(x = "total_bill", data=tips)

import pandas as pd
data=pd.read_csv('lb1/tips.csv')

#Общий счет (Total bill)
#Чаевые (Tips)
#Пол (Sex)
#Курящий (Smoker)
#День недели (Day)
#Время (Time)
#Количество (Size)

# Выводим первые 5 записей набора данных
data.head()
# Смотрим на количество имеющихся данных
len(data)
# линейная регрессия с скатером (рассеяние двумерного признака)
#ci - confidence interval (доверительный интервал)
plt.subplot(2, 2, 3)
sns.regplot(x='tip', y='total_bill', data=data, ci=None)

plt.subplot(2, 2, 4)
plt.hist(data['total_bill'])
# Whether to plot a gaussian kernel density estimate.
# Whether to draw a rugplot(коврик) on the support axis.
plt.figure()
sns.distplot(data['total_bill'], kde=False, rug=True)

sns.set_style('darkgrid')

plt.figure()
sns.boxplot(x='day', y='total_bill', data=data, palette='Blues')

# Добавляем параметры и сохраняем рисунок
#fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True))
#axes[0, 0].plot(x, y)
#axes[1, 1].scatter(x, y)
#subplots - Create a figure and a set of subplots

fig, ax= plt.subplots()
ax.scatter(data[data['sex']=='Male']['tip'], data[data['sex']=='Male']['total_bill'], color='blue', label='Муж')
ax.scatter(data[data['sex']=='Female']['tip'], data[data['sex']=='Female']['total_bill'], color='green', label='Жен')
ax.set_title('Общий счет и чаевые с распределением по полу', fontdict={'fontsize': 14, 'fontweight': 'bold'})
ax.set_ylabel('Общий счет')
ax.set_xlabel('Чаевые')
ax.legend()
plt.show()

#fig.savefig('tips.png', format='png', dpi=250)

# Отобразить результат
plt.show()
