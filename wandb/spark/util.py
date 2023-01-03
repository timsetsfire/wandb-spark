
from tempfile import TemporaryDirectory

def wandb_log_sparkml_model(sparkModelOrPipeline):
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

    config = wandb.config
    wandb.config = dict(**config, **model_config)
    wandb.run.update()

  with TemporaryDirectory() as tmp_dir:
    sparkModelOrPipeline.write.overwrite().save(f"file:/{tmp_dir.name}")
    artifact = wandb.Artifact(name = f"spark-pipeline-{wandb.run.id}", type = "model", metadata = model_config)
    artifact.add_dir(tmp_dir.name)
    wandb.log_artifact(artifact)
