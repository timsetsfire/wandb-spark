import wandb
import itertools
from pyspark import keyword_only
from typing import Any, Dict, Optional, TYPE_CHECKING
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import (
    HasLabelCol,
    HasPredictionCol,
    HasProbabilityCol,
    HasRawPredictionCol,
    HasFeaturesCol,
    HasWeightCol,
)
from pyspark.ml.evaluation import Evaluator, RegressionEvaluator, MulticlassClassificationEvaluator, BinaryClassificationEvaluator, MultilabelClassificationEvaluator, RankingEvaluator
class WandbEvaluator(Evaluator):   
  """
  Wrapper for pyspark Evaluators

  It is expected that the user will provide an Evaluators, and this wrapper will log metrics from said evaluator 
  to W&B.  
  """
  sparkMlEvaluator: Param[Evaluator] = Param(
        Params._dummy(),
        "sparkMlEvaluator",
        "evaluator from pyspark.ml.evaluation"        
    )

  wandbRun: Param[wandb.sdk.wandb_run.Run] = Param(
      Params._dummy(),
      "wandbRun",
      "wandb run.  Expects an already initialized run.  You should set this, or wandbRunKwargs, NOT BOTH"
  )

  wandbRunKwargs: Param[dict] = Param( 
      Params._dummy(), 
      "wandbRunKwargs", 
      "kwargs to be passed to wandb.init.  You should set this, or wandbRunId, NOT BOTH.  Setting this is useful when using with WandbCrossValdidator"
  )
  
  wandbRunId: Param[str] = Param(
      Params._dummy(),
      "wandbRunId",
      "wandb run id.  if not providing an intialized run to wandbRun, a run with id wandbRunId will be resumed"
  )
  
  metricPrefix: Param[str] = Param(
    Params._dummy(), 
    "metricPrefix", 
    "metric prefix for w&b run",
    typeConverter = TypeConverters.toString
  )
  
  wandbProjectName: Param[str] = Param(
    Params._dummy(), 
    "wandbProjectName", 
    "name of W&B project",
    typeConverter = TypeConverters.toString
  )
  
  labelValues: Param[list] = Param(
    Params._dummy(),
    "labelValues", 
    "for classification and multiclass classification, this is a list of values the label can assume\nIf provided Multiclass or Multilabel evaluator without labelValues, we'll figure it out from dataset passed through to evaluate.", 
  )

  _input_kwargs: Dict[str, Any]
  
  @keyword_only
  def __init__(self, *,               
               wandbRun: wandb.sdk.wandb_run.Run = None,
               sparkMlEvaluator: Evaluator = None,
               labelValues: list = None, 
               metricPrefix: str = None):
    
    super(Evaluator, self).__init__()
    
    self.metrics = {
      MultilabelClassificationEvaluator: ["subsetAccuracy", "accuracy",
                                          "hammingLoss", "precision", "recall",
                                          "f1Measure", "precisionByLabel", 
                                          "recallByLabel", "f1MeasureByLabel", 
                                          "microPrecision", "microRecall", "microF1Measure"],
      MulticlassClassificationEvaluator: ["f1", "accuracy", "weightedPrecision", 
                               "weightedRecall", "weightedTruePositiveRate", 
                               "weightedFalsePositiveRate", "weightedFMeasure", 
                               "truePositiveRateByLabel", "falsePositiveRateByLabel", 
                               "precisionByLabel", "recallByLabel", 
                               "fMeasureByLabel", "logLoss","hammingLoss"], 
      RegressionEvaluator: ["rmse", "mse", "r2", "mae", "var"], 
      BinaryClassificationEvaluator: ["areaUnderROC", "areaUnderPR"], 
      RankingEvaluator: ["meanAveragePrecision","meanAveragePrecisionAtK","precisionAtK","ndcgAtK","recallAtK"]
    }
    
    self._setDefault(labelValues=[], metricPrefix="eval/")
    kwargs = self._input_kwargs
    self._set(**kwargs)
    
    
  def setSparkMlEvaluator(self, value: Evaluator):
    self._set(sparkMlEvaluator=value)
  def setWandbRun(self, value: wandb.sdk.wandb_run.Run):
    self._set(wandbRunId=value.id)
    self._set(wandbProjectName=value.project)
    self._set(wandbRun=value)
  # def setWandbRunId(self, value: str):
  #   self._set(wandbRunId=value)
  #   project_name = self.getWandbProjectName()
  #   run = wandb.init(project = project_name, id = value, resume = "allow")
  #   self._set(wandbRun = run)
  def setWandbRunKwargs(self, value: dict): 
    self._set(wandbRunKwargs = value)
    run = wandb.init(**value)
    self._set(wandbRun = run)
  # def setWandbProjectName(self, value: str):
  #   self._set(wandbProjectName=value)
  def setMetricPrefix(self, value: str):
    self._set(metricPrefix=value)
  def setLabelValues(self, value: list):
    self._set(labelValues=value)
    
  def getSparkMlEvaluator(self):
    return self.getOrDefault(self.sparkMlEvaluator)
  def getWandbRun(self):
    return self.getOrDefault(self.wandbRun)
  def getWandbRunKwargs(self):
    return self.getOrDefault(self.wandbRunKwargs)
  def getWandbRunId(self):
    return self.getOrDefault(self.wandbRunId)
  def getWandbProjectName(self):
    return self.getOrDefault(self.wandbProjectName)
  def getMetricPrefix(self):
    return self.getOrDefault(self.metricPrefix)
  def getLabelValues(self):
    return self.getOrDefault(self.labelValues)
    
  def _evaluate(self, dataset):
    dataset.persist()
    metric_values = []
    labelValues = self.getLabelValues()
    sparkMlEvaluator = self.getSparkMlEvaluator()
    metricPrefix = self.getMetricPrefix()
    run = self.getWandbRun()
    evaluator_type = type(sparkMlEvaluator)
    if isinstance(sparkMlEvaluator, RankingEvaluator):
      metric_values.append( ("k", sparkMlEvaluator.getK()))      
    for metric in self.metrics[evaluator_type]:  
      if "ByLabel" in metric:
        if labelValues == []:
          print("no label_values for the target have been provided and will be determined by the dataset.  This could take some time")
          labelValues = [ r[sparkMlEvaluator.getLabelCol()] for r in dataset.select(sparkMlEvaluator.getLabelCol()).distinct().collect()]
          if isinstance(labelValues[0], list):
            merged = list(itertools.chain(*labelValues))
            labelValues = list(dict.fromkeys(merged).keys())           
          self.setLabelValues(labelValues)
        for label in labelValues:  
          out = sparkMlEvaluator.evaluate(dataset, {sparkMlEvaluator.metricLabel: label, sparkMlEvaluator.metricName: metric} )
          metric_values.append( (f"{metricPrefix}{metric}:{label}", out) )
      else:
        out = sparkMlEvaluator.evaluate(dataset, {sparkMlEvaluator.metricName: metric} )
        metric_values.append( (f"{metricPrefix}{metric}", out))
    run.log( dict(metric_values))
    config = [(f"{k.parent.split('_')[0]}.{k.name}", v) for k,v in sparkMlEvaluator.extractParamMap().items() if "metric" not in k.name]
    run.config.update( dict(config))
    ## already ran this, but am lazy :) 
    return_metric = sparkMlEvaluator.evaluate(dataset)
    dataset.unpersist()
    return return_metric

  def isLargerBetter(self):
      return self.getSparkMlEvaluator().isLargerBetter()
