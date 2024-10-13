import logging
import pandas as pd
import numpy as np
import mlflow

from src.evaluation import MSE,R2Score,RMSE
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

experiment_tracker = Client().activate_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluation(model:RegressorMixin, x_test:pd.DataFrame, y_test:pd.Series) -> Tuple[
    Annotated[float, "r2_score"], 
    Annotated[float, "rmse"]
]:
    try:
        prediction = model.predict(x_test) 
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("mse",mse)
        
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test,prediction)
        mlflow.log_metric("r2_score",r2_score)
        
        rmse_class = RMSE()
        rmse_score = rmse_class.calculate_score(y_test,prediction)
        mlflow.log_metric("rmse_score",rmse_score)
        
        return r2_score, rmse_score
    
    except Exception as e:
        logging.error(e)
        raise e
    
 
        

