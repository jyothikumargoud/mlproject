#import all the necessary librarys for mdoel training 
import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import eval_model

@dataclass
class ModelTrainerConfig:
        Trained_model_file_path = os.path.join("artifacts","model.pkl")
class ModelTrainer:
        def __init__(self):
                #model_trainer_config stores trained model file path
                self.model_trainer_config = ModelTrainerConfig()

        
        def initiate_model_trainig(self,train_array,test_array):
                try:
                        logging.info("training and test input data")
                        #seperating training and testing data from train and test arrays
                        X_train,y_train,X_test,y_test = (
                                train_array[:,:-1],
                                train_array[:,-1],
                                test_array[:,:-1],
                                test_array[:,-1]

                        )
                        # Models 
                        models = {
                                "Random Forest": RandomForestRegressor(),
                                "Decesion tree": DecisionTreeRegressor(),
                                "Gradient Booeting":GradientBoostingRegressor(),
                                "Linear Regression":LinearRegression(),
                                "K - Neighbour":KNeighborsRegressor(), 
                                "XG boost":XGBRegressor(),
                                "Cat boost Regreeeor":CatBoostRegressor(verbose =False),
                                "Ada boost regresssor":AdaBoostRegressor(),
                                
                        }


                        model_report: dict = eval_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, y=y_train, models=models)
                        # getting the best model score out of alla the models
                        best_model_score  = max(sorted(model_report.values()))

                        # getting the best model name out of all the models

                        best_model_name = list(model_report.keys())[
                                # go ti the index of best model score and get the model name
                                list(model_report.values()).index(best_model_score)
                                
                        ]
                        

                        best_model = models[best_model_name]
                        if best_model_score < 0.6:
                                raise CustomException("no best model found")
                        logging.info(f'Best model found on training and testing dataset')


                        save_object(
                                file_path=self.model_trainer_config.Trained_model_file_path,
                                obj=best_model
                        )
                        predicted = best_model.predict(X_test)
                        r2score = r2_score(y_test,predicted)
                        return r2score
                
                        
                except Exception as e:
                        raise CustomException(e,sys)

                        



