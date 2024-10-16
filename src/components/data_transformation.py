from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from logger import logging
from src.exception_handling import CustomException
import sys
import pandas as pd
import os
from src.utils import save_preprocessor

@dataclass
class data_transformation_config:
  preprocesser_obj_path = os.path.join('Artifacts','preprocessor.pkl')

class data_transformation:
  def __init__(self):
    self.data_transformation_cfg = data_transformation_config()

  def transformer(self):

    try:

      
      one_hot_category = ['parental_level_of_education','race_ethnicity']

      cat_transformer = ColumnTransformer(
        transformers = [

          # ('Imputer',SimpleImputer(strategy="most_frequent"),one_hot_features),
          ('One Hot Encoding',OneHotEncoder(sparse_output=False),one_hot_category),
        ],remainder = 'passthrough'
      )

      cat_transformer.set_output(transform='pandas')

      return cat_transformer
    except Exception as e:
      raise CustomException(e,sys)
  
  def intiate_transformation(self,train_data_path,test_data_path):
    try:
      logging.info('Started reading the train and test data')
      train_df = pd.read_csv(train_data_path)
      test_df = pd.read_csv(test_data_path)

      logging.info('Succesfully Read the data')
      logging.info('Transformation started')


      train_df['avg_mark'] = (train_df['math_score'] + train_df['reading_score'] + train_df['writing_score'])//3
      test_df['avg_mark'] = (test_df['math_score'] + test_df['reading_score'] + test_df['writing_score'])//3

      train_df.drop(['math_score','reading_score','writing_score'],axis=1,inplace=True)
      test_df.drop(['math_score','reading_score','writing_score'],axis=1,inplace=True)

      
      train_df.loc[train_df['avg_mark'] < 50, 'avg_mark'] = 0
      train_df.loc[train_df['avg_mark']>=50,'avg_mark'] = 1
      test_df.loc[test_df['avg_mark'] >= 50, 'avg_mark'] = 1
      test_df.loc[test_df['avg_mark'] < 50, 'avg_mark'] = 0


      



      #Label Encoding
      label_encoding = ['lunch','gender','test_preparation_course']
      le = LabelEncoder()
      for i in label_encoding:
        train_df[i] = le.fit_transform(train_df[i])
        test_df[i] = le.transform(test_df[i])
        
      transformer = self.transformer()
     

      train_df = transformer.fit_transform(train_df)
      test_df = transformer.transform(test_df)
      logging.info('Transformation Done')
      save_preprocessor(transformer,self.data_transformation_cfg.preprocesser_obj_path)
      
      return train_df,test_df

    except Exception as e:
      raise CustomException(e,sys)
