# Artificial Neural Network to predict demand for electric energy in Poland

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os

# Import training set
path_train = r'C:\Users\Dorota Nowak\Desktop\Energia\train'
all_files_train = glob.glob(os.path.join(path_train, "*.csv"))

df_from_each_file = (pd.read_csv(f, sep=';') for f in all_files_train)
train = pd.concat(df_from_each_file, ignore_index=True)
prognoza_train = train.iloc[:, 2].values
train.drop("Dobowa prognoza zapotrzebowania KSE", axis=1, inplace=True)
train.rename(columns={'Godz.': 'hr'}, inplace=True)
train.replace('2A', '2', inplace=True)

# Import test set
path_test = r'C:\Users\Dorota Nowak\Desktop\Energia\test'
all_files_test = glob.glob(os.path.join(path_test, "*.csv"))

df_from_each_file = (pd.read_csv(f, sep=';') for f in all_files_test)
test = pd.concat(df_from_each_file, ignore_index=True)
prognoza_test = test.iloc[:, 2].values
test.drop("Dobowa prognoza zapotrzebowania KSE", axis=1, inplace=True)
test.rename(columns={'Godz.': 'hr'}, inplace=True)
test.replace('2A', '2', inplace=True)

# Encoding the time in training and test set
hours_train = train['hr'].astype(float)
train['hr_sin'] = np.sin(hours_train * (2. * np.pi / 24))
train['hr_cos'] = np.cos(hours_train * (2. * np.pi / 24))
# train.drop('hr.', axis=1, inplace=True)

hours_test = test['hr'].astype(float)
test['hr_sin'] = np.sin(hours_test * (2. * np.pi / 24))
test['hr_cos'] = np.cos(hours_test * (2. * np.pi / 24))
# test.drop('hr.', axis=1, inplace=True)
# test.plot.scatter('hr_sin', 'hr_cos').set_aspect('equal');
# plt.show()

# Encoding the days of the week and the day of the year
import datetime

day_sinX = []
day_cosX = []
date_sinX = []
date_cosX = []
for data in train['Data']:
    year = int(str(data)[:4])
    month = int(str(data)[4:6])
    day = int(str(data)[6:8])
    day_sinX.append(np.sin(datetime.datetime(year, month, day).weekday() * (2. * np.pi / 7)))
    day_cosX.append(np.cos(datetime.datetime(year, month, day).weekday() * (2. * np.pi / 7)))
    date_sinX.append(
        np.sin((datetime.date(year, month, day) - datetime.date(year, 1, 1)).days + 1) * (2. * np.pi / 366))
    date_cosX.append(
        np.cos((datetime.date(year, month, day) - datetime.date(year, 1, 1)).days + 1) * (2. * np.pi / 366))

train['day_sin'] = day_sinX
train['day_cos'] = day_cosX
train['date_sin'] = date_sinX
train['date_cos'] = date_cosX
# train.plot.scatter('date_sin', 'date_cos').set_aspect('equal')
# plt.show()

day_siny = []
day_cosy = []
date_siny = []
date_cosy = []
for data in test['Data']:
    year = int(str(data)[:4])
    month = int(str(data)[4:6])
    day = int(str(data)[6:8])
    day_siny.append(np.sin(datetime.datetime(year, month, day).weekday() * (2. * np.pi / 7)))
    day_cosy.append(np.cos(datetime.datetime(year, month, day).weekday() * (2. * np.pi / 7)))
    date_siny.append(
        np.sin((datetime.date(year, month, day) - datetime.date(year, 1, 1)).days + 1) * (2. * np.pi / 366))
    date_cosy.append(
        np.cos((datetime.date(year, month, day) - datetime.date(year, 1, 1)).days + 1) * (2. * np.pi / 366))

test['day_sin'] = day_siny
test['day_cos'] = day_cosy
test['date_sin'] = date_siny
test['date_cos'] = date_cosy

# Creating X_train, X_test, y_train, y_test
X_train = train.iloc[:, [3, 4, 5, 6, 7, 8]].values
y_train = train.iloc[:, 2].values
max1 = 0
for i, yy in enumerate(y_train):
    if '-' in yy:
        y_train[i] = prognoza_train[i]

for i, yy in enumerate(y_train):
    if type(yy) is str:
        y_train[i] = y_train[i].replace(',', '.')
    y_train[i] = float(y_train[i])
    if y_train[i] > max1:
        max1 = y_train[i]

X_test = test.iloc[:, [3, 4, 5, 6, 7, 8]].values
y_test = test.iloc[:, 2].values
for i, yy in enumerate(y_test):
    if '-' in yy:
        y_test[i] = prognoza_test[i]

for i, yy in enumerate(y_test):
    if type(yy) is str:
        y_test[i] = y_test[i].replace(',', '.')
    y_test[i] = float(y_test[i])
    if y_test[i] > max1:
        max1 = y_test[i]

# Normalizing y_train and y_test
y_train = y_train / max1
y_test = y_test / max1

# Zapotrzebowanie na energię godzinę i dwie godziny przed i dzień przed o tej samej godzinie
y_train_one_hour_before = y_train
y_train_one_hour_before = np.insert(y_train_one_hour_before, 0, 16012.93 / max1)
y_train_one_hour_before = y_train_one_hour_before[:-1]

y_train_two_hours_before = y_train_one_hour_before
y_train_two_hours_before = np.insert(y_train_two_hours_before, 0, 16561.8 / max1)
y_train_two_hours_before = y_train_two_hours_before[:-1]

y_train_one_day_before = y_train
december14 = pd.read_csv('20141231.csv', sep=';')
december14 = december14.iloc[:, 3].values
for i, yy in enumerate(december14):
    if type(yy) is str:
        december14[i] = december14[i].replace(',', '.')
    december14[i] = float(december14[i])
december14 = december14 / max1
y_train_one_day_before = np.r_[december14, y_train_one_day_before]
y_train_one_day_before = y_train_one_day_before[:-24]

y_test_one_hour_before = y_test
y_test_one_hour_before = np.insert(y_test_one_hour_before, 0, 15968.388 / max1)
y_test_one_hour_before = y_test_one_hour_before[:-1]

y_test_two_hours_before = y_test_one_hour_before
y_test_two_hours_before = np.insert(y_test_two_hours_before, 0, 15504.775 / max1)
y_test_two_hours_before = y_test_two_hours_before[:-1]

y_test_one_day_before = y_test
december17 = y_train[-24:]
y_test_one_day_before = np.r_[december17, y_test_one_day_before]
y_test_one_day_before = y_test_one_day_before[:-24]

X_train = np.c_[X_train, y_train_one_hour_before, y_train_two_hours_before, y_train_one_day_before]
X_test = np.c_[X_test, y_test_one_hour_before, y_test_two_hours_before, y_test_one_day_before]
X_train = X_train.astype(float)
X_test = X_test.astype(float)


# ANN
def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoid_derv(s):
    return s * (1 - s)


def error(pred, real):
    return pred - real.astype(float)


np.random.seed(40)


class MyNN:
    def __init__(self, x, y):
        self.x = x
        neurons = 13
        self.lr = 0.01
        ip_dim = x.shape[1]  # 9
        op_dim = 24

        self.w1 = np.random.randn(ip_dim, neurons)
        self.b1 = np.zeros((1, neurons))
        self.w2 = np.random.randn(neurons, op_dim)
        self.b2 = np.zeros((1, op_dim))
        self.y = y

    def feedforward(self, i):
        z1 = np.dot(self.x[i:i + 1], self.w1) + self.b1
        self.a1 = sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(z2)

    def backprop(self, i):
        loss = error(self.a2, self.y[i:i + 24])
        sum_loss = np.sum(loss**2)
        print('Error :', sum_loss)
        errors.append(sum_loss)

        der_2 = sigmoid_derv(self.a2)
        a2_delta = 2*loss*der_2
        d_weights2 = np.dot(self.a1.T, a2_delta)

        der_1 = sigmoid_derv(self.a1)
        a1_delta = np.dot(a2_delta, self.w2.T)*der_1
        d_weights1 = np.dot(self.x[i:i+1].T, a1_delta)

        # update the weights with the derivative (slope) of the loss function
        self.w1 -=self.lr* d_weights1
        self.w2 -=self.lr* d_weights2


    def predict(self, data):
        self.x = data
        self.feedforward(0)
        return self.a2


# y_train = y_train.reshape(-1, 1)
# print(X_train.shape)
# print(y_train.shape[1])
model = MyNN(X_train, y_train)

errors = []
error_in_epoch = []
epochs = 100
for x in range(epochs):
    for i in range(26284):
        model.feedforward(i)
        model.backprop(i)
    error_in_epoch.append(sum(errors))
    errors = []

plt.plot(error_in_epoch)
plt.xlabel('epoch')
plt.ylabel('total error sum')
plt.show()

predicted_y_test = []
days_of_the_week_error = [0]*7
days_of_the_year_error = [0]*365

def get_acc(x, y):

    acc = 0
    for i, xx in enumerate(x):
        if (i >= x.shape[0]-24):
            print(acc)
            return acc / (x.shape[0]-24) * 100
        xx = xx.reshape((1, 9))
        s = model.predict(xx)  # wektor 24 licz
        yy = y[i:i + 24]
        predicted_y_test.append(s)
        dif = abs(s - yy) / yy  # wektor 24 liczb
        dif = np.sum(dif) / 24
        days_of_the_week_error[int(np.floor(i/24)%7)] +=dif
        days_of_the_year_error[int(i/24)] +=dif
        acc += dif


#print("Training MAPE : ", get_acc(X_train, np.array(y_train)))
print("Test MAPE : ", get_acc(X_test, np.array(y_test)))



label = ['pn','wt','sr','cz','pt','sb','nd']
x = np.arange(len(label))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(x,days_of_the_week_error,0.5, align='center')
ax.set_xticks(x)
ax.set_xticklabels(label)
fig.show()

x1 = np.arange(len(days_of_the_year_error))
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.bar(x1,days_of_the_year_error,0.5, align='center')
fig1.show()



#MAPE 5.43 dla 100 epok i lr = 0.01
