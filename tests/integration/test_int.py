from DiamondPricePredictor.logger import logging
from DiamondPricePredictor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME='data_ingestion_pipeline'
try:
    logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logging.exception(e)
    raise e