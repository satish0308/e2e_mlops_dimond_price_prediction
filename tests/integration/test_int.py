from DiamondPricePredictor.logger import logging
from DiamondPricePredictor.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

def test_integration():
    STAGE_NAME = 'data_ingestion_pipeline'
    try:
        logging.info(f">>>>>> Test for stage {STAGE_NAME} started <<<<<<")
        run_data_ingestion_pipeline()
        logging.info(f">>>>>> Test for stage {STAGE_NAME} completed successfully <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(f"Error during test for stage {STAGE_NAME}")
        raise e

def run_data_ingestion_pipeline():
    logging.info("Running data ingestion pipeline...")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    # Add assertions or checks here to verify the expected behavior
    logging.info("Data ingestion pipeline completed successfully.")
