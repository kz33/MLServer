from mlserver import ModelSettings
from .errors import InvalidMLLibFormat
from pyspark.mllib.classification import LogisticRegressionModel, NaiveBayesModel, SVMModel
from pyspark.mllib.clustering import KMeansModel
from pyspark.mllib.regression import IsotonicRegressionModel, LinearRegressionModel, LassoModel, RidgeRegressionModel
from pyspark.mllib.tree import DecisionTreeModel, GradientBoostedTreesModel, RandomForestModel


async def get_mllib_load(settings: ModelSettings):
    if not settings.parameters:
        raise InvalidMLLibFormat(settings.name)

    mllib_format = settings.parameters.format

    if not mllib_format:
        raise InvalidMLLibFormat(settings.name)

    model_dict = {
        "LogisticRegression": LogisticRegressionModel.load,
        "NaiveBayes": NaiveBayesModel.load,
        "SVM": SVMModel.load,
        "KMeans": KMeansModel.load,
        "IsotonicRegression": IsotonicRegressionModel.load,
        "LinearRegression": LinearRegressionModel.load,
        "Lasso": LassoModel.load,
        "RidgeRegression": RidgeRegressionModel.load,
        "DecisionTree": DecisionTreeModel.load,
        "GradientBoostedTrees": GradientBoostedTreesModel.load,
        "RandomForest": RandomForestModel.load
    }

    if mllib_format not in model_dict:
        raise InvalidMLLibFormat(settings.name)

    return model_dict[mllib_format]
