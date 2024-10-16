import os
import dill
from src.logger import logging
from src.exception_handling import CustomException
import sys
def save_preprocessor(preprocessor_obj,location_to_save):

  try:
    os.makedirs(os.path.dirname(location_to_save),exist_ok=True)

    with open(location_to_save,'wb') as file_location:
      dill.dump(obj = preprocessor_obj,file = file_location)
    logging.info('Saved preprocessor object')
    

  except Exception as e:
    raise CustomException(e,sys)


