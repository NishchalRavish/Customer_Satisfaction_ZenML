import logging 
import pandas as pd

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
    
class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """
    def train(self,x_train,y_train,**kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train,y_train)
        return reg