import wandb
import xgboost as xgb
import easydict
import json 
import pprint
import os

class  WandbXGBoostSparkCallback(xgb.callback.TrainingCallback):
  def findkeys(node, kv):
    if isinstance(node, list):
        for i in node:
            for x in findkeys(i, kv):
               yield x
    elif isinstance(node, dict):
        if kv in node:
            yield node[kv]
        for j in node.values():
            for x in findkeys(j, kv):
                yield x
  def __init__(self, project_name, group = None, use_in_cv = False, cv_params = None):
    self.run = None
    self.project_name = project_name
    self.group = group
    self.use_in_cv = use_in_cv
    self.cv_params = cv_params
  def before_training(self, model):
    wandb.require('service')
    job_type = "cv" if self.use_in_cv else None
    model_conf = json.loads(model.save_config())
    # self.run.config.update(model_conf, allow_val_change = True)
    self.run = wandb.init(
      project = self.project_name,
      job_type = job_type, 
      group = self.group,
      config = model_conf,
      settings=wandb.Settings(start_method="fork"))
    return model
  def after_iteration(self, model, epoch, evals_log):
    for key, item in evals_log.items():
      for metric, value in item.items():
        if isinstance(value, list):
          for i, v in enumerate(value):
            self.run.log({key: v}, step = i)
        else:
          self.run.log({key: item})
  def after_training(self, model):
    # model_conf = json.loads(model.save_config())
    # self.run.config.update(model_conf, allow_val_change = True)
    if self.use_in_cv:
      self.run.log({"cv_params": self.cv_params})
      for param in self.cv_params:
          try:
            if param == "n_estimators":
              param_value = int(model.num_boosted_rounds())
            else:
              param_value = list(findkeys(model_conf, param)).pop(0)
          except Exception as e:
            param_value = str(e)
          self.run.log({param: param_value})   
    print(model.attributes())
    for k, v in model.attributes().items():
      self.run.log( {k: float(v)})
    self.run.log({"num_boosted_rounds": int(model.num_boosted_rounds())})
    print(model.num_boosted_rounds())
    model.save_model(f"xgb-{self.run.id}.bin")
    model_artifact = wandb.Artifact(name = f"xgb-{self.run.id}", type = "model", metadata = model_conf)
    model_artifact.add_file(f"xgb-{self.run.id}.bin")
    self.run.log_artifact(model_artifact)
    self.run.finish()
    return model