import os
from box.exceptions import BoxValueError
import yaml
from src.CNNClassifierKidneyDiseases import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64


# Ensure annotation force the datatype to be as i declared!
# as it convrted each var as class in a dict, then checks it 
#{'x': <class 'int'>, 'y': <class 'int'>, 'return': <class 'int'>}
"""
def add(x: int, y: int) -> int:
    return x + y

add("3", "4")

this function is run without any problem although i declared that
x, y must be numbers !

thats why we use @ensure_annotations
"""

@ensure_annotations
def read_yaml(path_to_yaml:Path)->ConfigBox:
    """
    Reads yaml file and returns

    Args:
        path_to_yaml : Path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox : ConfigBox Type

    
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e 
    
    """
    config = Box(yaml.safe_load(open("config.yaml")))

    🔁 Step-by-Step:
    open("config.yaml")
    → Opens your config file (in YAML format)

    yaml.safe_load(...)
    → Parses the YAML file into a Python dict

    Box(...)
    → Converts that dict into a Box object so you can access values like:

    config.model.name
    config.training.batch_size
    
    """


# ------------------------------------------- #


@ensure_annotations
def create_directories(path_to_directories:list, verbose=True):
    """ 
        Create a list of directories

        Args:
            path_to_directories(list)L list of path of directories
            ignore_log(bool , optional) : ignore if multiple dirs is to be created
    
    """

    for path in path_to_directories:
        os.makedirs(path , exist_ok=True)
        if verbose:
            logger.info(f"Created Directory at : {path}")



#-----------------------------------------------#


@ensure_annotations
def save_json(path:Path , data:dict):
    """
    save json data

    """

    with open(path, "w")as f:
        json.dump(data , f , indent=4)
        # dump : add seralized datatype like dict,list into a file of json format

    logger.info(f"json File Saved Successfully at: {path}")



@ensure_annotations
def load_json(path:Path)->ConfigBox:
    """
        Loading Json File
    """

    with open(path)as f:
        content = json.load(f)
        logger.info(f"Json file is Loaded Successfully from {path}")
        return ConfigBox(content)
    


# ---------------------------------------- #

@ensure_annotations
def load_bin(path:Path)->Any:
    """
        Loading Binary data
    """

    data = joblib.load(path)
    logger.info(f"Binary File is loaded successfully : {path}")
    return data


@ensure_annotations
def get_size(path:Path)->str:
    """
        get size in KB
    """
    size = round(os.path.getsize(path)/1024)
    return f"{size} KB"



def decode_image(imgstring,filename):
    img_data = base64.b64decode(imgstring)
    with open(filename,"wb") as f:
        f.write(img_data)
        f.close()

"""
    the img is passed as string in APIs, Request 
    so we decode it from string into a normal img in pixels
"""


def encode_image(cropped_img):
    with open(cropped_img,"rb") as f:
        return base64.b64encode(f.read())
    
    