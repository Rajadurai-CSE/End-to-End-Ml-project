from sklearn.model_selection import train_test_split
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
import os
import sys
from src.exception_handling import CustomException
from data_transformation import data_transformation
from model_trainer import best_model_finder

@dataclass
class data_ingestion_config:
  train_data_path : str = os.path.join('Artifacts' ,'train.csv') # artifacts will be present inside the src
  test_data_path : str = os.path.join('Artifacts' , 'test.csv')
  # raw_data_path : str 
  raw_data_path:str = os.path.join('Artifacts','raw.csv')


class data_ingestion:
  def __init__(self):
    self.ingestion_config = data_ingestion_config()

  def ingest(self):
    logging.info('Data Ingestion Intiated')
    try:
      df = pd.read_csv(r'D:/Ml_Project/notebook/data/stud.csv')
      logging.info('Read the dataset')
      #ingesting the data
      os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
     
      #save raw data
      df.to_csv(self.ingestion_config.raw_data_path,index=False)
      logging.info('Saved raw data')
      logging.info('Intiated the train test split')
      train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
      train_set.to_csv(self.ingestion_config.train_data_path,index=False)
      test_set.to_csv(self.ingestion_config.test_data_path,index=False)
      logging.info('Saved the train test split data')
    except Exception as e:
      raise CustomException(e,sys)
    

if __name__ == '__main__':
  obj = data_ingestion()
  obj.ingest()
  dt = data_transformation()
  x_train,x_test =  dt.intiate_transformation(obj.ingestion_config.train_data_path,obj.ingestion_config.test_data_path)
  print(best_model_finder(train_df=x_train,test_df=x_test).pick_best_model())




