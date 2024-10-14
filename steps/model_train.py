import logging
import pandas as pd
import mlflow
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from .config import ModelNameConfig

from src.model_dev import(
    LinearRegressionModel,
    RandomForestModel,
    LightGBMModel,
    HyperparameterTuner,
    XGBoostModel
)

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(x_train:pd.DataFrame, x_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series, config:ModelNameConfig) -> RegressorMixin:
    """
    Trains the ML model on the ingested data
    
    Args:
        x_train:pd.DataFrame
        x_test:pd.DataFrame
        y_train:pd.Series
        y_test:pd.Series
    """
    try:
        model = None
        tuner = None
        
        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()        
        
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
            
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
            
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            
        else:
            raise ValueError("Model not supported")
        
        tuner = HyperparameterTuner(model,x_train,y_train,x_test,y_test)
        
        if config.fine_tuning:
            best_params = tuner.optimize()
            trained_model = model.train(x_train,y_train,**best_params)
        else:
            trained_model = model.train(x_train,y_train)
        
        return trained_model

    except Exception as e:
        logging.error(e)
        raise e
    
