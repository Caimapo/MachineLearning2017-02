import numpy   as np
import pandas  as pd
import os

from hyperopt import Trials, STATUS_OK, tpe
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from sklearn.model_selection import train_test_split



def data():
    base_path = os.path.join('data', 'sign-language-mnist')
    train = pd.read_csv(os.path.join(base_path, 'sign_mnist_train.csv'))
    # read train set
    y_train = train['label'].as_matrix()
    x_train = train.iloc[:, 1:].as_matrix()
    # split train set into train and validation set
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)
    
    x_train = x_train.astype('float64') / 255
    x_test = x_test.astype('float64') / 255
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense({{choice([30, 45, 60])}}, input_dim=x_train.shape[1], activation={{choice(['relu', 'elu'])}}, kernel_initializer='uniform'))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense({{choice([30, 45, 60])}}, activation={{choice(['relu', 'elu'])}}, kernel_initializer='uniform'))
    model.add(Dropout({{uniform(0, 1)}}))
    
    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense({{choice([30, 45, 60])}}, input_dim=x_train.shape[1], activation={{choice(['relu', 'elu'])}}, kernel_initializer='uniform'))
        model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense(25, activation='softmax', kernel_initializer='uniform'))
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})
    
    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=100,
              verbose=0,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

trial_obj = Trials()
best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=75,
                                        trials=trial_obj)
X_train, Y_train, X_test, Y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)
with open("trial.json", "a") as myfile:
    myfile.write(trial_obj.trials.__str__())
