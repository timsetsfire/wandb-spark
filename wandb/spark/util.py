
from tempfile import TemporaryDirectory
from pyspark.ml import Pipeline, PipelineModel
import wandb

def log_model(sparkModelOrPipeline, path = None):
  model_config = {}
  if isinstance(sparkModelOrPipeline, PipelineModel):
    for stage in sparkModelOrPipeline.stages:
      m = stage.extractParamMap()
      str_stage = f"{stage.__class__.__module__}.{stage.__class__.__name__}"
      model_config[str_stage] = {}
      model_config[str_stage]["uid"] = stage.uid
      for k,v in m.items():
        model_config[str_stage][k.name] = v
  else:
    m = sparkModelOrPipeline.extractParamMap()
    str_stage = f"{stage.__class__.__module__}.{stage.__class__.__name__}"
    model_config[str_stage] = {}
    model_config[str_stage]["uid"] = stage.uid
    for k,v in m.items():
      model_config[str_stage][k.name] = v

    # wandb.config.update(model_config)
  artifact = wandb.Artifact(name = f"spark-pipeline-{wandb.run.id}", type = "model", metadata = model_config)
  if path is None:
    with TemporaryDirectory() as tmp_dir:
      sparkModelOrPipeline.write().overwrite().save(f"file:///{tmp_dir}")
      artifact.add_dir(tmp_dir)
      wandb.log_artifact(artifact)
  else: 
    artifact.add_dir(path)
    wandb.log_artifact(artifact)