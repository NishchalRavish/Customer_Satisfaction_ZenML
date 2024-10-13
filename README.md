# Customer_Satisfaction_ZenML
Customer_Satisfaction_ZenML

Initialize zenml - zenml init 

Start local zenml server - zenml up

Stop local zenml server - zenml down

Describe zenml stack - zenml stack describe

List zenml stack - zenml stack list

Install zenml integrations - zenml integration install mlflow -y

Register mlflow tracker - zenml experiment-tracker register mlflow_tracker --flavor=mlflow

Register mlflow model deployer - zenml model-deployer register mlflow --flavor=mlflow

Register mlflow stack - zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set

Get mlflow uri - print(Client().active_stack.experiment_tracker.get_tracking_uri()) (Run the output in terminal to get link of localhost for mlflow server)
We get - file:/Users/nishchal/Library/Application Support/zenml/local_stores/e7e12e54-4b4d-4692-bdb3-551c445814d0/mlruns