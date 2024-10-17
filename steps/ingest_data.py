import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingest the data from the data path
    """
    def __init__(self) -> None:
        """
        Initialize the data ingestion class
        """
        pass
        
    def get_data(self) -> pd.DataFrame:
        df = pd.read_csv('data/olist_customers_dataset.csv')
        return df
        

@step
def ingest_data() -> pd.DataFrame:
    """
    Ingest the data from the data path
    
    Args:
        None
    Returns:
        df: pd.DataFrame
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
    
    