import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
  #exc_info gives detailed information about the error
  # exc_tb --> the traceback object that holds the information about the call stack at the point where the exception has occured
  #Frame that we extract file_name and error line
  _,_,exc_tb = error_detail.exc_info()
  file_name = exc_tb.tb_frame.f_code.co_filename
  error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
  return error_message


class CustomException(Exception):
  def __init__(self,error_message,error_detail :sys):
    super().__init__(error_message)
    self.error_message = error_message_detail(error_message,error_detail)

  def __str__(self) -> str:
    return self.error_message
  

# if __name__ == '__main__':
#   try:
#     a = 1 / 0
#   except Exception as e:
#      logging.info('wghrong')
#      raise CustomException(e,sys)

