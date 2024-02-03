from DiamondPricePredictor.logger import logging
from DiamondPricePredictor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from DiamondPricePredictor.pipeline.stage_02_data_transofrm_model import ModelCreationTrainingPipeline



STAGE_NAME = "Data Ingestion stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e




STAGE_NAME = "Prepare base model"
try: 
   logging.info(f"*******************")
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = ModelCreationTrainingPipeline()
   prepare_base_model.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logging.exception(e)
        raise e