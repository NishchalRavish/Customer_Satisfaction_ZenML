import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from src.model_dev import(
    LinearRegressionModel
)

@step
def train_model(x_train:pd.DataFrame, y_train:pd.Series) -> RegressorMixin:
    """
    Trains the ML model on the ingested data
    
    Args:
        x_train:pd.DataFrame
        x_test:pd.DataFrame
        y_train:pd.Series
        y_test:pd.Series
    """
    try:
        model = LinearRegressionModel()
        trained_model = model.train(x_train,y_train)
        
        return trained_model

    except Exception as e:
        logging.error(e)
        raise e
    
