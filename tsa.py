from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
dateparse=lambda dates:pd.datetime.strptime(dates,'%m') 
data=pd.read_excel('Demandv1.1 (1).xlsx',sheet_name=1,parse_dates=['Month DD Raised'],index_col='Month DD Raised',date_parser=dateparse)
data.head()
dummy=pd.get_dummies(data['Skill Group'],drop_first=True)
#dummy.columns
data.drop(['Skill Group'],axis=1,inplace=True)
data=pd.concat([data,dummy],axis=1)
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
value=data.values
data.fillna(0)
missing_values=((data.isnull().sum()/len(data))*100).sort_values(ascending=False)
encoder = LabelEncoder()
value[:,1] = encoder.fit_transform(value[:,1])
value[:,2]=encoder.fit_transform(value[:,2])
value[:,3]=encoder.fit_transform(value[:,3])
value[:,4]=encoder.fit_transform(value[:,4])
value[:,5]=encoder.fit_transform(value[:,5])
value = value.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(value)
from pandas import DataFrame
from pandas import concat
# frame as supervised learning
reframed = series_to_supervised(scaled, 1,1)
reframed=reframed[['var1(t-1)','var2(t-1)', 'var3(t-1)', 'var4(t-1)', 'var5(t-1)','var6(t-1)','var1(t)','var2(t)', 'var3(t)', 'var4(t)', 'var5(t)', 'var6(t)']]
#split into train and test sets
values = reframed.values
n_train_hours=int(len(reframed)*0.80)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:,0:6], train[:,6:]
test_X, test_y = test[:,0:6], test[:,6:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(6))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=16, batch_size=52, validation_data=(test_X, test_y), verbose=2, shuffle=False)
from math import sqrt
from sklearn.metrics import mean_squared_error
from numpy import concatenate
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
# invert scaling for forecast
inv_yhat = concatenate((yhat[:,-6:], test_X[:,:]), axis=1)
#inv_yhat = scaler.inverse_transform(inv_yhat)
#inv_yhat = inv_yhat[:,:]
# invert scaling for actual
#test_y = test_y.reshape((len(test_y),12))
#inv_y = concatenate((test_y, test_X[:, :]), axis=1)
#inv_y = scaler.inverse_transform(inv_y)
#inv_y = inv_y[:,:]
# calculate RMSE
rmse = sqrt(mean_squared_error(test_y,yhat))
print('Test RMSE: %.3f' % rmse)