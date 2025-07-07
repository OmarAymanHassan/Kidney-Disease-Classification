from src.CNNClassifierKidneyDiseases.constants import *
from src.CNNClassifierKidneyDiseases.utils.common import read_yaml , create_directories
from src.CNNClassifierKidneyDiseases.entity.config_entity import DataIngestionConfig
from src.CNNClassifierKidneyDiseases.entity.config_entity import PrepareBaseModelConfig
from src.CNNClassifierKidneyDiseases.entity.config_entity import TrainingConfig
from src.CNNClassifierKidneyDiseases.entity.config_entity import EvaluationConfig
import os 


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
    

    
    # Config for Model Training
    def get_training_config(self)->TrainingConfig:

        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir,"kidney-ct-scan-image")

        create_directories([training.root_dir])


        training_config = TrainingConfig(

            root_dir=training.root_dir,
            trained_model_path=Path(training.training_model_path),
            updated_base_model=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmented=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE

        )

        return training_config

# Model Evaluation 
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(

            path_of_model="artifacts/training/model.pth",
            training_data="artifacts/data_ingestion/kidney-ct-scan-image",
            mlflow_uri="https://dagshub.com/omarkhadrawy10/Kidney-Disease-Classification.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE

            
        )

        return eval_config


# Configuration For Model Training 


    


    