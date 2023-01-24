import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from GeneticFeatures.GeneticFeatureGenerator import *
from GeneticFeatures.Node import *

# random forest, mlp, gradient_boosting regression models

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LogisticRegression

class tester:
    def __init__(self, dataset, target, generator, nofeatures, nosplits, maxsplitsize, verbose = False, test_size = 0.2, random_state = 42):
        self.dataset = dataset
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.dataset, self.target, test_size = self.test_size, random_state = self.random_state)   
        self.multifeature_generator = MultiFeatureGenerator(self.X_train, self.y_train, generator, nofeatures, nosplits, maxsplitsize, verbose)

        self.trees = None
        self.new_X_train, self.new_X_test = self.create_data()

        self.X_scaler = StandardScaler()

        self.new_X_scaler = StandardScaler()

        self.target_scaler = StandardScaler()

        self.X_scaler.fit(self.X_train)
        self.new_X_scaler.fit(self.new_X_train)
        self.target_scaler.fit(self.y_train.reshape(-1, 1))

        self.X_train = self.X_scaler.transform(self.X_train)
        self.X_test = self.X_scaler.transform(self.X_test)
        self.new_X_train = self.new_X_scaler.transform(self.new_X_train)
        self.new_X_test = self.new_X_scaler.transform(self.new_X_test)
        self.y_train = self.target_scaler.transform(self.y_train.reshape(-1, 1)).reshape(-1)
        self.y_test = self.target_scaler.transform(self.y_test.reshape(-1, 1)).reshape(-1)

        self.tests = pd.DataFrame(columns = ['Model', 'MSE', 'R2', 'MAE','NEW_MSE', 'NEW_R2', 'NEW_MAE'])
    
    def create_data(self):
        self.trees = [i for i in self.multifeature_generator]
        new_cols_test_data = [t(self.X_test) for t in self.trees]
        new_cols_train_data = [t(self.X_train) for t in self.trees]

        new_X_test = np.array(new_cols_test_data).T
        new_X_train = np.array(new_cols_train_data).T

        new_train_data = np.concatenate((self.X_train, new_X_train), axis = 1)
        new_test_data = np.concatenate((self.X_test, new_X_test), axis = 1)

        return new_train_data, new_test_data

    
    def test_model(self, model, kargs, notests, show_iterations):
        acc_results = np.zeros(6)
        for i in range(notests):

            oldmodel = model(**kargs)
            newmodel = model(**kargs)
            oldmodel.fit(self.X_train, self.y_train)
            newmodel.fit(self.new_X_train, self.y_train)

            old_pred = oldmodel.predict(self.X_test)
            new_pred = newmodel.predict(self.new_X_test)

            old_pred = self.target_scaler.inverse_transform(old_pred.reshape(-1, 1))
            new_pred = self.target_scaler.inverse_transform(new_pred.reshape(-1, 1))

            y_test = self.target_scaler.inverse_transform(self.y_test.reshape(-1, 1))
            
            old_mse = mean_squared_error(y_test, old_pred)
            old_r2 = r2_score(y_test, old_pred)
            old_mae = mean_absolute_error(y_test, old_pred)

            new_mse = mean_squared_error(y_test, new_pred)
            new_r2 = r2_score(y_test, new_pred)
            new_mae = mean_absolute_error(y_test, new_pred)

            acc_results += np.array([old_mse, old_r2, old_mae, new_mse, new_r2, new_mae])

            if (i+1) % show_iterations == 0:
                print(f'Iteration {i} - MSE {old_mse}, R2 {old_r2}, MAE {old_mae}, NEW_MSE {new_mse}, NEW_R2 {new_r2}, NEW_MAE {new_mae}')

        acc_results /= notests
        results = {'Model': oldmodel.__class__.__name__, 'MSE': acc_results[0], 'R2': acc_results[1], 'MAE': acc_results[2], 'NEW_MSE': acc_results[3], 'NEW_R2': acc_results[4], 'NEW_MAE': acc_results[5]}
        self.tests = pd.concat([self.tests, pd.DataFrame(results, index = [0])], ignore_index = True)

    def test_models(self, models = None, kargs = None, notests = 1, show_iterations = 5):
        # if models == None:
        models, kargs = self.get_default_models()

        for model, karg in zip(models, kargs):
            self.test_model(model, karg, notests, show_iterations)

    def get_tests(self):
        return self.tests

    def get_default_models(self):
        models =  [
            RandomForestRegressor, 
            # MLPRegressor, 
            SGDRegressor,
            GradientBoostingRegressor]
        kargs = [
            {'n_estimators': 100, 'max_depth': 10},
            # {'hidden_layer_sizes': (64, 32, 16), 'max_iter': 1000},
            {'max_iter': 1000, 'tol': 1e-3},
            {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'learning_rate': 0.1}
        ]
        return models, kargs

    
