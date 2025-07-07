from pathlib import Path
import dagshub
from dagshub import dagshub_logger, init

import mlflow

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# -----------


def dagshub_info():
    dagshub.init(repo_owner='omarkhadrawy10', repo_name='Kidney-Disease-Classification', mlflow=True)

    with mlflow.start_run():
        mlflow.log_param('parameter name', 'value')
        mlflow.log_metric('metric name', 1)