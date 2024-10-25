from src.utils import save_model
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


from src.utils import evaluate_models
from src.logger import logging
@dataclass
class model_config:
  model_obj_path = os.path.join('Artifacts','model.pkl')


def models():
   models_list = {
      "KNN Regressor" : KNeighborsRegressor(),
         "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
             

                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
   params = {
      
      "KNN Regressor" :{
         'n_neighbors' : [3,4,5,6,7,8,9,10],
         'weights' : ['uniform','distance'],
         'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']
         
      },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
   }
   return models_list,params
   


class best_model_finder:
    def __init__(self,train_df,test_df):
      self.model_path = model_config().model_obj_path
      self.train_df = train_df
      self.test_df = test_df
      self.models_list,self.params = models()
      self.threshold = 0.75
      
    def pick_best_model(self):
        x_train = self.train_df.iloc[:,:-1]
        y_train = self.train_df.iloc[:,-1]
        x_test = self.test_df.iloc[:,:-1]
        y_test = self.test_df.iloc[:,-1]
        report = evaluate_models(x_train,y_train,x_test,y_test,self.models_list,self.params)
        # best_model = max(zip(list(report.values())[1], report.keys()))[1]
        best_model = max(report, key=lambda k: report[k][1])
        best_model = report[best_model]


        if best_model[0] >= self.threshold and best_model[1]>=  self.threshold:
           logging.info('Best Model Found')
           
           save_model(best_model[2],self.model_path)
           
           return f"Best Model Test R2 Score {best_model[1]}"
        
        else:
           return "Best Model not found"



        #After getting report pick a model whose r2 training score and test score is minimum of 0.75

        #Sort function
        #Pick the key with max r2_test score 
        # check if train and test score is above threshold
        #if available then save the model
        #else return no best model found





    
    


  
  




