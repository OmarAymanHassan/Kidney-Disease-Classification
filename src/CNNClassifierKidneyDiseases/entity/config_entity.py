from dataclasses import dataclass
from pathlib import Path




@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    source_url:str
    local_data_files:Path
    unzip_dir:Path



# Model Configuration




@dataclass(frozen=True)
# frozen=true : dont add anything by the user 
# so those vars will become constant and we cant add any var

class PrepareBaseModelConfig:
    root_dir:Path
    base_model_path:Path
    updated_base_model_path:Path
    params_image_size:list
    params_learning_rate:float
    params_including_top:int
    params_weight:str
    params_classes:int
    params_model_name:str
