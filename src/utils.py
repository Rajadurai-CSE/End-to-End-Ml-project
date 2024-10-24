import os
import dill
from src.logger import logging
from src.exception_handling import CustomException
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def save_preprocessor(preprocessor_obj,location_to_save):

  try:
    os.makedirs(os.path.dirname(location_to_save),exist_ok=True)

    with open(location_to_save,'wb') as file_location:
      dill.dump(obj = preprocessor_obj,file = file_location)
    logging.info('Saved preprocessor object')
    

  except Exception as e:
    raise CustomException(e,sys)
  


def evaluate_models(x_train,y_train,x_test,y_test,models,params):

  report = {}
  models_list = list(models.values())
  models_param = list(params.values())
  for i in range(len(models_list)):
    gsv = GridSearchCV(cv=5,estimator = models_list[i],param_grid=models_param[i])
    gsv.fit(x_train,y_train)

    y_train_pred = gsv.best_estimator_.predict(x_train)
    y_test_pred = gsv.best_estimator_.predict(x_test)

    train_r2score = r2_score(y_true=y_train,y_pred=y_train_pred)
    test_r2score = r2_score(y_true=y_test,y_pred=y_test_pred)

    report[list(models.keys())[i]] = [train_r2score,test_r2score,gsv.best_estimator_]

  return report







  
def save_model(model_obj,location_to_save):
  try:
    os.makedirs(os.path.dirname(location_to_save),exist_ok=True)

    with open(location_to_save,'wb') as file_location:
      dill.dump(model_obj,file_location)

    logging.info('Model Pickle saved successfully')

  except Exception as e:
    raise CustomException(e,sys)




def load_object(path):
  try:
    read_obj = open(path,'rb')
    obj = dill.load(read_obj)
    read_obj.close()
    return obj
  
  except Exception as e:
    raise CustomException(e,sys)
