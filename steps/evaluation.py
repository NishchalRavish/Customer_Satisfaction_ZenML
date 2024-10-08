import logging
from zenml import step

@step
def evaluate_model(df:pd.DataFrame) -> None:
    """
    Evaluates the model on ingested data
    
    Args:
        df: the ingested data
    """
    pass