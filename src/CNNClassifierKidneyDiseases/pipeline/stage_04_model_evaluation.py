from src.CNNClassifierKidneyDiseases.config.configuration import ConfigurationManager
from src.CNNClassifierKidneyDiseases.components.model_evaluation import Evaluation
from src.CNNClassifierKidneyDiseases import logger

STAGE_NAME=  "Evaluation Stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f"***************************")
        logger.info(f">>>>>>>>>> stage {STAGE_NAME} Started <<<<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>> Stage {STAGE_NAME} Completed Successfully <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e