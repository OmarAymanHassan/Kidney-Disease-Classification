from src.CNNClassifierKidneyDiseases.constants import *
from src.CNNClassifierKidneyDiseases.utils.common import read_yaml , create_directories
from src.CNNClassifierKidneyDiseases.entity.config_entity import DataIngestionConfig
from src.CNNClassifierKidneyDiseases.entity.config_entity import PrepareBaseModelConfig

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH , params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifact_root])


    def get_data_ingestion_config(self)->DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
        root_dir = config.root_dir,
        source_url = config.source_url,
        local_data_files = config.local_data_files,
        unzip_dir = config.unzip_dir
        )

        return data_ingestion_config


# Model Configuration 


    def get_prepare_base_model(self)->PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size= self.params.IMAGE_SIZE,
            params_learning_rate= self.params.LEARNING_RATE,
            params_classes= self.params.CLASSES,
            params_weight= self.params.WEIGHTS,
            params_including_top= self.params.INCLUDING_TOP,
            params_model_name=self.params.MODEL_NAME
                    )
        return prepare_base_model_config
    





