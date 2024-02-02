import os
from pathlib import Path
from DiamondPricePredictor.constants import *
from DiamondPricePredictor.utils.common import read_yaml, create_directories
from DiamondPricePredictor.entity.config_entity import DataIngestionConfig



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file_train=config.local_data_file_train,
            local_data_file_test=config.local_data_file_test,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config