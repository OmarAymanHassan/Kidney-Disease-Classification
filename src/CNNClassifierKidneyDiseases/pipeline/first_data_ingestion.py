
from src.CNNClassifierKidneyDiseases.entity.config_entity import DataIngestionConfig
from src.CNNClassifierKidneyDiseases.config.configuration import ConfigurationManager
from src.CNNClassifierKidneyDiseases.components.data_ingestion import DataIngestion
from src.CNNClassifierKidneyDiseases import logger



STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            # it automatic take the args from the constant variales that assigned
            #those assignment : CONFIG_FILE_PATH , PARAMS_FILE_PATH
            data_ingestion = DataIngestion(config= data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extraxt_zip_file()
        except Exception as e:
            raise e
        

if __name__ == "__main__":
    try:
        logger.info(f">>>>>>>>>> Satge {STAGE_NAME} Started <<<<<<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed <<<<<<<<<<")

    except Exception as e:
        logger.exception(e)
        raise e