# demo.py
from HF.pipline.training_pipeline import TrainPipeline
from HF.entity.config_entity import TrainingPipelineConfig

if __name__ == "__main__":
    # Define the configuration for the training pipeline
    training_pipeline_config = TrainingPipelineConfig(pipeline_name="Heart Failure Prediction Pipeline")

    # Pass the config when creating the TrainPipeline object
    obj = TrainPipeline(training_pipeline_config=training_pipeline_config)

    # Run the pipeline
    obj.run()
