from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import sys
from src.logger import logging
from src.exception_handling import CustomException

import pandas as pd
import os
from src.utils import save_preprocessor

@dataclass
class data_transformation_config:
  preprocesser_obj_path = os.path.join('Artifacts','preprocessor.pkl')

class data_transformation:
  def __init__(self):
    self.data_transformation_cfg = data_transformation_config()


  def transformer(self,cat_features,num_features):

    try:

      #For categorical transformation

      cat_pipeline = Pipeline(
        [
          ('Mode Imputer',SimpleImputer(strategy="most_frequent")),
          ('One Hot Encoder',OneHotEncoder(sparse_output=False))
        ]
      )

     #For numerical transformation

      num_pipeline = Pipeline(
        [
          ('Median Imputer',SimpleImputer(strategy="median")),
          ('Standard Scaling',StandardScaler())
        ]
      )
      
      #Merging two
      
      column_transformer = ColumnTransformer(
        transformers = [
          ('cat transformer',cat_pipeline,cat_features),
          ('num transformer',num_pipeline,num_features)
        ],remainder = 'passthrough'
      )

      column_transformer.set_output(transform='pandas')

      # transformer.set_output('pandas')

      return column_transformer
    except Exception as e:
      raise CustomException(e,sys)
  
  def intiate_transformation(self,train_data_path,test_data_path):
    try:
      logging.info('Started reading the train and test data')
      train_df = pd.read_csv(train_data_path)
      test_df = pd.read_csv(test_data_path)

      logging.info('Succesfully Read the data')
      logging.info('Transformation started')

      train_target = train_df['math_score']
      test_target = test_df['math_score']

      train_df = train_df.drop(['math_score'], axis=1)
      test_df = test_df.drop(['math_score'], axis=1)


      cat_features = train_df.select_dtypes(include='object').columns
      num_features = train_df.select_dtypes(exclude='object').columns

      #Convert cat_features values to lower case
      for i in cat_features:
        train_df[i] = train_df[i].str.lower()
        test_df[i] = test_df[i].str.lower()

      
      transformer = self.transformer(cat_features,num_features)
      x_train_transformed = transformer.fit_transform(train_df)
    
      x_test_transformed = transformer.transform(test_df)

      x_train = pd.concat([x_train_transformed,train_target],axis=1)
      x_test = pd.concat([x_test_transformed,test_target],axis=1)

      logging.info('Transformation Done')
      
      save_preprocessor(transformer,self.data_transformation_cfg.preprocesser_obj_path)
      
      return x_train,x_test

    except Exception as e:
      raise CustomException(e,sys)
