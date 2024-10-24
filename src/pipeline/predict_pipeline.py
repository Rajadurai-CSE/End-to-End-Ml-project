
import pandas as pd
import os
from src.utils import load_object



preprocessor_path = os.path.join('Artifacts','preprocessor.pkl')
model_path = os.path.join('Artifacts','Model.pkl')

def predict(user_df):
  preprocessor = load_object(preprocessor_path)
  transformed_user_df = preprocessor.transform(user_df)
  model = load_object(model_path)
  prediction = model.predict(transformed_user_df)
  return prediction



def userdata_df(gender:str,race_ethnicity:str,parental_level_of_education:str,lunch:str,test_preparation_course:str,reading_score:float,writing_score:float):

  user_dict = {
    'gender':gender,
    'race_ethnicity':race_ethnicity,
    'parental_level_of_education':parental_level_of_education,
    'lunch':lunch,
    'test_preparation_course':test_preparation_course,'reading_score':reading_score,
    'writing_score':writing_score
  }

  user_df = pd.DataFrame([user_dict])
  return user_df


