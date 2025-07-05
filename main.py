from src.CNNClassifierKidneyDiseases import logger
from src.CNNClassifierKidneyDiseases.pipeline.first_data_ingestion import DataIngestionTrainingPipeline


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


