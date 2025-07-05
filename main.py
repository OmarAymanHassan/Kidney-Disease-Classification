from src.CNNClassifierKidneyDiseases import logger
from src.CNNClassifierKidneyDiseases.pipeline.first_data_ingestion import DataIngestionTrainingPipeline
from src.CNNClassifierKidneyDiseases.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

logger.info("Welcome to the Custom logger")
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>>>>>> Satge {STAGE_NAME} Started <<<<<<<<<<")
    data_ingestion_one = DataIngestionTrainingPipeline()
    data_ingestion_one.main()
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

except Exception as e:
    logger.exception(e)
    raise e


##### MODEL CONFIG #####

STAGE_NAME = "Prepare Base Model"


try:
    logger.info(f"***************************")
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed Successfully <<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e


