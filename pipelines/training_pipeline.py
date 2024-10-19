from zenml.pipelines import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.model_train import train_model

@pipeline(enable_cache=False)
def train_pipeline(ingest_data,clean_data,train_model,evaluation):
    """
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
        
    Returns:
        mse: float
        rmse: float
    """
    df = ingest_data()
    x_train,x_test,y_train,y_test = clean_data(df)
    model = train_model(x_train,x_test,y_train,y_test)
    r2_score,rmse_score = evaluation(model,x_test,y_test)