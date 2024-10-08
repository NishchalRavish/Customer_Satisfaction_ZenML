import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingest the data from the data path
    """
    def __init__(self,data_path:str):
        """
        Args: 
            data_path: path to the data
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Ingesting the data from the data path and returns a Dataframe
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

# Zenml data ingestion step    
@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    Ingest the data from the data path
    
    Args:
        data_path: path to data
    Returns:
        pd.DataFrame: Dataframe of the data
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
    
    