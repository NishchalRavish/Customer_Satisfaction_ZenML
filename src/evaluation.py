import logging
import numpy as np

from abc import ABC,abstractmethod
from sklearn.metrics import mean_squared_error,r2_score

class Evaluation(ABC):
    """
    Abstract class that defines the strategy for evaluating model performance
    """
    @abstractmethod
    def calculate_score(self, y_true:np.ndarray,y_pred:np.ndarray) -> float:
        pass
    
class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_score(self, y_true:np.ndarray, y_pred:np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            mse:float
        """
        try:
            logging.info("Entered the calculate_score mehtod for MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("The MSE value is + str(mse)")
            return mse
        except Exception as e:
            logging.error(e)
            raise e
        
        
class R2Score(Evaluation):
    """
    Evaluation Strategy that uses R2 Score
    """    
    def calculate_score(self, y_true:np.ndarray,y_pred:np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            r2_score:float
        """        
        try:
            logging.info("Entered the calculate_score mehtod for R2 Score")
            r2 = r2_score(y_true,y_pred)
            logging.info("The R2 Score value is + str(r2)")
            return r2
        except Exception as e:
            logging.error(e)
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation Strategy that uses RMSE Score
    """    
    def calculate_score(self, y_true:np.ndarray,y_pred:np.ndarray) -> float:
        """
        Args:
            y_true: np.ndarray
            y_pred: np.ndarray
        Returns:
            rmse:float
        """        
        try:
            logging.info("Entered the calculate_score mehtod for RMSE Score")
            rmse = np.sqrt(mean_squared_error(y_true,y_pred))
            logging.info("The RMSE Score value is + str(rmse)")
            return rmse
        except Exception as e:
            logging.error(e)
            raise e