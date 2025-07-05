import os
import urllib.request as request
import zipfile
from src.CNNClassifierKidneyDiseases import logger
from src.CNNClassifierKidneyDiseases.utils.common import get_size
import gdown
from src.CNNClassifierKidneyDiseases.entity.config_entity import DataIngestionConfig

# the first thing that this class take is the config from the 
# DataClassConfiguration 
class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config


    def download_file(self)->str:
        """
            Fetch Data from the url
        """

        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_files
            os.makedirs(self.config.root_dir , exist_ok=True)
            logger.info(f"Downloading Data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix ='https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id , zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e

    def extraxt_zip_file(self):

        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok = True)
        with zipfile.ZipFile(self.config.local_data_files , "r")as zip_ref:
            zip_ref.extractall(unzip_path)



