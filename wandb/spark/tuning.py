import os
import sys
import itertools
from multiprocessing.pool import ThreadPool

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    overload,
    TYPE_CHECKING,
)

import numpy as np

from pyspark.ml.tuning import * 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark import keyword_only, since, SparkContext, inheritable_thread_target
from pyspark.ml import Estimator, Transformer, Model
from pyspark.ml.common import inherit_doc, _py2java, _java2py
from pyspark.ml.evaluation import Evaluator, JavaEvaluator
from pyspark.ml.param import Params, Param, TypeConverters
from pyspark.ml.param.shared import HasCollectSubModels, HasParallelism, HasSeed
from pyspark.ml.util import (
    DefaultParamsReader,
    DefaultParamsWriter,
    MetaAlgorithmReadWrite,
    MLReadable,
    MLReader,
    MLWritable,
    MLWriter,
    JavaMLReader,
    JavaMLWriter,
)
from pyspark.ml.wrapper import JavaParams, JavaEstimator, JavaWrapper
from pyspark.sql.functions import col, lit, rand, UserDefinedFunction
from pyspark.sql.types import BooleanType

from pyspark.sql.dataframe import DataFrame

if TYPE_CHECKING:
    from pyspark.ml._typing import ParamMap
    from py4j.java_gateway import JavaObject
    from py4j.java_collections import JavaArray

__all__ = [
    "ParamGridBuilder",
    "CrossValidator",
    "CrossValidatorModel",
    "TrainValidationSplit",
    "TrainValidationSplitModel",
]

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

from pyspark.ml.pipeline import PipelineModel 

   
def wandbParallelFitTasks(
    est: Estimator,
    train: DataFrame,
    eva: Evaluator,
    validation: DataFrame,
    epm: Sequence["ParamMap"],
    collectSubModel: bool,
) -> List[Callable[[], Tuple[int, float, Transformer]]]:

    modelIter = est.fitMultiple(train, epm)

    def singleTask() -> Tuple[int, float, Transformer]:
        index, model = next(modelIter)
        # TODO: duplicate evaluator to take extra params from input
        #  Note: Supporting tuning params in evaluator need update method
        #  `MetaAlgorithmReadWrite.getAllNestedStages`, make it return
        #  all nested stages and evaluators
        try:
        #   eva.setWandbRunId(epm[index][eva.getParam("wandbRunId")])
            eva.setWandbRunKwargs(epm[index][eva.getParam("wandbRunKwargs")])
        except Exception as e:
          print(e)
          raise Exception("failing")
        run = eva.getWandbRun()
        if isinstance(model, PipelineModel):
            conf = []
            for stage in model.stages:
                params = stage.extractParamMap()
                conf.extend( [(f"{k.parent.split('_')[0]}.{k.name}", v) for k,v in params.items()] )
        else:
            params = model.extractParamMap()
            conf = [(f"{k.parent.split('_')[0]}.{k.name}", v) for k,v in params.items()]
        conf = dict(conf)
        run.config.update(conf)
        eva.evaluate(model.transform(train, epm[index]) , {"metricPrefix": "train/"})
        metric = eva.evaluate(model.transform(validation, epm[index]))
        wandb.finish()
        return index, metric, model if collectSubModel else None

    return [singleTask] * len(epm)

class WandbCrossValidator(CrossValidator): 

    _input_kwargs: Dict[str, Any]

    @keyword_only
    def __init__(
        self,
        *,
        estimator: Optional[Estimator] = None,
        estimatorParamMaps: Optional[List["ParamMap"]] = None,
        evaluator: Optional[Evaluator] = None,
        numFolds: int = 3,
        seed: Optional[int] = None,
        parallelism: int = 1,
        collectSubModels: bool = False,
        foldCol: str = "",
    ) -> None:
        """
        __init__(self, \\*, estimator=None, estimatorParamMaps=None, evaluator=None, numFolds=3,\
                 seed=None, parallelism=1, collectSubModels=False, foldCol="")
        """
        super(CrossValidator, self).__init__()
        self._setDefault(parallelism=1)
        kwargs = self._input_kwargs
        self._set(**kwargs)


    def _fit(self, dataset: DataFrame) -> "CrossValidatorModel":
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        metrics_all = [[0.0] * numModels for i in range(nFolds)]

        pool = ThreadPool(processes=min(self.getParallelism(), numModels))
        subModels = None
        collectSubModelsParam = self.getCollectSubModels()
        if collectSubModelsParam:
            subModels = [[None for j in range(numModels)] for i in range(nFolds)]

        datasets = self._kFold(dataset)
        for i in range(nFolds):
            validation = datasets[i][1].cache()
            train = datasets[i][0].cache()

            tasks = map(
                inheritable_thread_target,
                wandbParallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam),
            )
            for j, metric, subModel in pool.imap_unordered(lambda f: f(), tasks):
                metrics_all[i][j] = metric
                if collectSubModelsParam:
                    assert subModels is not None
                    subModels[i][j] = subModel

            validation.unpersist()
            train.unpersist()

        metrics, std_metrics = CrossValidator._gen_avg_and_std_metrics(metrics_all)

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
        bestModel = est.fit(dataset, epm[bestIndex])
        return self._copyValues(
            CrossValidatorModel(bestModel, metrics, cast(List[List[Model]], subModels), std_metrics)
        )
