import os
from pathlib import Path
import logging

logging.basicConfig(
      level=logging.INFO,
      format='[%(asctime)s]:%(message)s:'

 )

project_name ="diamondPricePredictor"

list_of_files=[
    ".github/workflows/.gitkeep",
    ".github/workflows/ci.yaml",
    ".github/workflows/product-release.yaml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/logger.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/unit/unit_test.py",
    "tests/integration/__init__.py",
    "tests/integration/Integration_test.py",
    "init_setup.sh",     
     "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiments/experiments.ipynb",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "requirements_dev.txt",
    "setup.py",
    "research/trails.ipynb",
    "templates/index.html",
    "main.py",
    "kaggle.json"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    file_dir,file_name=os.path.split(filepath)
    if file_dir!="":
        os.makedirs(file_dir,exist_ok=True)
        logging.info (f"creating the file Directory{file_dir} for the file name :{file_name} ")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open (filepath,'w') as f:
            pass
            logging.info(f'creating empty file :{file_name}')
    else:
        logging.info(f"{file_name} already exist")