import os,sys

import pandas as pd

#import mlflow.sklearn 
from customerchurn.exceptions.exception import CustomerChurnException
from customerchurn.logging.logger import logging
from customerchurn.constant.training_pipeline import TARGET_COLUMN
from customerchurn.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,DataValidationArtifact
from customerchurn.entity.config_entity import ModelTrainerConfig
from customerchurn.utils.main_utils.utils import load_numpy_array_data,evaluate_model
from customerchurn.utils.ml_utils.metrics.classification_metric import get_classification_score
from customerchurn.utils.main_utils.utils import save_object,load_object
from customerchurn.utils.ml_utils.model.estimator import CustomerChurnModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
    
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import mlflow

class ModelTrainer:
    
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact,
                 data_validation_artifact:DataValidationArtifact  
                 ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise CustomerChurnException(e,sys)

    def track_mlflow(self,best_model_pipeline,test_metric):
        with mlflow.start_run():
            f1_score = test_metric.f1_score 
            precision_score = test_metric.f1_score
            recall_score = test_metric.recall_score

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(best_model_pipeline,"model")


    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree" : DecisionTreeClassifier(),
                "Gradiant Boosting" : GradientBoostingClassifier(verbose=1),
                "Logistic Regression" : LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
                
            }
        
            params={
                "Decision Tree":{ 
                    'criterion':['gini','entropy','lop_loss'],
                    #'splitter':['best','random'],
                    #'max_features':['sqrt','log2'],
                    },
                "Random Forest":{
                    #'critersion':['gini','entropy','log_loss'],
                    #'max_features':['sqrt','log2',None],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "Gradiant Boosting":{
                    #'loss':['log_loss','exponential'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.85,0.9],
                    #'critersion':['squared_error','friedman_mse'],
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256]
                    
                },
                "Logistic Regression":{},
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }
            
            preprocessor = load_object(self.data_transformation_artifact.transformation_object_file_path)

            #preform preprocesses 
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            #apply SMOTE
            
            smote = SMOTE(random_state=42)
            X_train_balanced,y_train_balanced = smote.fit_resample(X_train_transformed,y_train)
            
            
            
            model_report, best_model_pipeline = evaluate_model(
                X_train=X_train_balanced,
                y_train=y_train_balanced,
                X_test=X_test_transformed,
                y_test=y_test,
                models=models,
                param=params,
                preprocessor=None
            )
 
            y_train_pred = best_model_pipeline.predict(X_train_transformed)
            y_test_pred = best_model_pipeline.predict(X_test_transformed)

            train_metrics = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            test_metrics = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            ## Track mlflow 
            self.track_mlflow(best_model_pipeline,train_metrics)

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            save_object(self.model_trainer_config.trained_model_file_path, obj=best_model_pipeline)

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metrics,
                test_metric_artifact=test_metrics
            )

        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    def initiat_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            X_train = train_df.drop(columns=[TARGET_COLUMN])
            y_train = train_df[TARGET_COLUMN]

            X_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise CustomerChurnException(e, sys)