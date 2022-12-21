import wandb
import easydict
import json 
import pprint
import os
from xgboost.callback import TrainingCallback

class  WandbSparkXGBCallback(TrainingCallback):
  def __init__(self, **kwargs):
    self.run = None
    self.kwargs = kwargs
  def before_training(self, model):
#     model_conf = json.loads(model.save_config())
    # self.run.config.update(model_conf, allow_val_change = True)
    self.run = wandb.init(**self.kwargs)
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
    for k, v in model.attributes().items():
      self.run.log( {k: float(v)})
    self.run.log({"num_boosted_rounds": int(model.num_boosted_rounds())})
    self.run.finish()
    return model