import logging 
import pandas as pd
import optuna 
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract base class for all ML models
    """
    @abstractmethod
    def train(self,x_train,y_train):
        """
        Trains the model on the given data
        
        Args:
            x_train: Training data
            y_train: Target data
        """
        pass
    
    @abstractmethod
    def optimize(self,trial,x_train,y_train,x_test,y_test):
        """
        Optimizes the hyperparameters of the model
        
        Args:
            trial: Optuna trial object
            x_train: Training data
            y_train: Target data
            x_test: Testing data
            y_test: Testing target
        """
        pass
    
class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """
    def train(self,x_train,y_train,**kwargs):
        reg = RandomForestRegressor(**kwargs)
        reg.fit(x_train,y_train)
        return reg      
    
    def optimize(self,trial,x_train,y_train,x_test,y_test):
        n_estimators = trial.suggest_int("n_estimators",1,200)    
        max_depth = trial.suggest_int("max_depth",1,20)    
        min_samples_split = trial.suggest_int("min_samples_split",2,20)
        reg = self.train(x_train,y_train,n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)
        return reg.score(x_test,y_test)
    
class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """
    def train(self,x_train,y_train,**kwargs):
        reg = LGBMRegressor(**kwargs)
        reg.fit(x_train,y_train)
        return reg      
    
    def optimize(self,trial,x_train,y_train,x_test,y_test):
        n_estimators = trial.suggest_int("n_estimators",1,200)    
        max_depth = trial.suggest_int("max_depth",1,20)    
        learning_rate = trial.suggest_uniform("learning_rate",0.01,0.99)
        reg = self.train(x_train,y_train,n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
        return reg.score(x_test,y_test) 
 

class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """
    def train(self,x_train,y_train,**kwargs):
        reg = xgb.XGBRegressor(**kwargs)
        reg.fit(x_train,y_train)
        return reg      
    
    def optimize(self,trial,x_train,y_train,x_test,y_test):
        n_estimators = trial.suggest_int("n_estimators",1,200)    
        max_depth = trial.suggest_int("max_depth",1,30)    
        learning_rate = trial.suggest_uniform("learning_rate",1e-7,10.0)
        reg = self.train(x_train,y_train,n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate)
        return reg.score(x_test,y_test) 
    
class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """
    def train(self,x_train,y_train,**kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train,y_train)
        return reg
    
    def optimize(self,x_train,y_train,x_test,y_test):
        reg=self.train(x_train,y_train)
        return reg.score(x_test,y_test)
    
class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses model strategy to perform tuning.
    """
    def __init__(self,model,x_train,y_train,x_test,y_test):
        self.model=model,
        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test=y_test
        
    def optimize(self,n_trails=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.model.optimize(trial, self.x_train,self.y_train.self.x_test,self.y_test), n_trials=n_trails)
        return study.best_trial.params
        