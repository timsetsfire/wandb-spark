


def wandb_log_sparkml_model(sparkModelOrPipeline, path):
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

  sparkModelOrPipeline.write.overwrite().save(path)
  artifact = wandb.Artifact(name = "spark-pipeline", type = "model", metadata = model_config)
  artifact.add_dir(path)
  wandb.log_artifact(artifact)