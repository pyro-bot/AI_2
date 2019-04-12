import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def desc_data(data: pd.DataFrame):
    print(data.info())
    print(f'\nОбъектов: {len(data)}')
    print(f'\nПропусков:')
    print(data.isna().sum())
    print()
    print(data.head())

### tips ###

#о ресторане
data = pd.read_csv('../dataset/tips.csv')
#Общий счет (Total bill)
#Чаевые (Tip)
#Пол (Sex)
#Курящий (Smoker)
#День недели (Day)
#Время (Time)
#Количество (Size)

# анализ и описание данных
desc_data(data)

# диаграммы
plt.figure()
sns.set_style('darkgrid')

# курящих и некурящих компаний
plt.subplot(2, 2, 1)
sns.countplot(x='size', hue='smoker', data=data, palette='Blues')

# чаевые от курящих и некурящих
plt.subplot(2, 2, 2)
for smoker in ['No', 'Yes']:
    sns.kdeplot(data[data['smoker'] == smoker]['tip'], shade=True, legend=True, label=smoker)

# динамика заработка в течение недели
#print(data['day'].unique())
#print(data['time'].unique())
day_bill = []
for day in ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']:
    for time in ['Breakfast', 'Lunch', 'Dinner', 'Supper']:
        total_bill = data[
                (data['day'] == day) & (data['time'] == time)
                ]['total_bill']
        day_bill.append(total_bill.sum())
plt.subplot(2, 2, 3)
plt.plot(day_bill, 'o-')

# счет если платит мужчина или женщина
plt.subplot(2, 2, 4)
sns.boxplot(x='sex', y='total_bill', data=data, palette='Blues')



###
data = pd.read_csv('../dataset/premier_league.csv')
desc_data(data)
plt.figure()
#home_team away_team home_goals away_goals result season

# количество игр по командам
plt.subplot(2, 4, 1)
data['home_team'].value_counts().add(
        data['away_team'].value_counts(), fill_value=0
        ).plot(kind='bar', color='#4884af')

# тренд голов по сезонам
#print(data['season'].unique())
data['goals'] = data['home_goals'] + data['away_goals']
data['year'] = data['season'].apply(lambda season: int(season[-4:]))
regdata = data.groupby('year')['goals'].agg(np.sum)
plt.subplot(2, 4, 2)
plt.xlim(regdata.index.values[0], regdata.index.values[-1])
sns.regplot(x=regdata.index.values, y=regdata.values, ci=None)

# распределение голов в матче
plt.subplot(2, 4, 3)
sns.boxplot(x='season', y='goals', data=data, palette='Blues')
plt.xticks(rotation='vertical')

# количество клубов в сезоне
X = []
for season in data['season'].unique():
    sdata = data[data['season'] == season]
    X.append(len(np.union1d(sdata['home_team'].unique(), sdata['away_team'].unique())))
plt.subplot(2, 4, 4)
#print(X)
#print(len(data[data['season'] == '2006-2007']['home_team'].unique()))
sns.violinplot(x=X)

# гистограмма побед и порожений клуба
#print(data['result'].unique())
team = 'Arsenal'
params = [['H', 'А', 'blue'], ['A', 'H', 'red']]
plt.subplot(2, 4, 5)
#for
teamwin = data[
        (data['home_team'] == team) & (data['result'] == params[0][0]) |
        (data['away_team'] == team) & (data['result'] == params[0][1])
        ]['year'].value_counts().sort_index()
teamwin.plot.bar(width=1, alpha=0.5, color=params[0][2])
teamlose = data[
        (data['home_team'] == team) & (data['result'] == params[1][0]) |
        (data['away_team'] == team) & (data['result'] == params[1][1])
        ]['year'].value_counts().sort_index()
teamlose.plot.bar(width=1, alpha=0.5, color=params[1][2])

# распределение счета
plt.subplot(2, 4, 6)
sns.scatterplot(x="home_goals", y='away_goals', data=data, alpha=0.1)

# интерполяция вероятности выиграть у клуба
def kernel_reg(X, Xi, Yi: np.array, h):
    if len(np.shape(Xi)) != 1 or len(np.shape(Yi)) != 1 or len(Xi) != len(Yi):
        return []

    return [kernel_call(x, Xi, Yi, h) for x in X]

def kernel_call(x, Xi, Yi, h):
    W = np.array([kernel_wrap(xi-x, h) for xi in Xi])
    Y = sum(Yi * W) / sum(W);
    return Y

def kernel_wrap(d, h):
    #w = 1 if abs(d) <= h else 0
    w = 1/h * kernel(d/h);
    return w

def kernel(d):
    #гаусово
    raw_w = 1/np.sqrt(2*np.pi) * np.exp(-d**2/2); #std norm dist
    return raw_w

cumtl = teamlose.cumsum()
X = np.linspace(cumtl.index.values[0], cumtl.index.values[-1], 50)
plt.subplot(2, 4, 7)
plt.plot(X, kernel_reg(X, cumtl.index.values, cumtl.values, 2))

#тепловая диаграмма количества игр друг с другом (или join?)
teams = np.union1d(data['home_team'].unique(), data['away_team'].unique())
heatmatrix = pd.DataFrame(data=0, index=teams, columns=teams)
for team in teams:
    for index, value in data[data['away_team'] == team]['home_team'].value_counts().iteritems():
        heatmatrix.loc[team, index] += value
    for index, value in data[data['home_team'] == team]['away_team'].value_counts().iteritems():
        heatmatrix.loc[team, index] += value
plt.subplot(2, 4, 8)
sns.heatmap(heatmatrix, xticklabels=True, yticklabels=True)

plt.show()
