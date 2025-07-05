import os
from pathlib import Path
import logging 

# Logging strings 

logging.basicConfig(level=logging.INFO , format="[%(asctime)s: %(message)s:]")
#  it prints out the logging time `asci time` + message itslef

project_name = "CNN-Classifier-Kidney-Diseases"

# Create the template of the project: Creating some files and folders

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html"
]



for filepath in list_of_files:
    filepath = Path(filepath)
    # we use path: because it handles the / or \ and see what is the os ! 
    # if its linux or windows or whatever, it deals with it and provide the right path
    file_dir , file_name = os.path.split(filepath)
    #file_dir : all the absolute path before the current file 
    # file_name : the current file name 
    """  
     filepath = "data/raw/info.txt"
    => file_dir = "data/raw"
    => file_name = "info.txt"

      
        """
    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating Directories: {file_dir} for the files: {file_name}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) ==0):
        with open(filepath , "w") as f:
            logging.info(f" Creating Empty File: {filepath}")

    else:
        logging.info(f"{file_name} is already exists.")





