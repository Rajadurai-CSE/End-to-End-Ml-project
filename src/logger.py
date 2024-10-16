#Custom log Function

import logging
import os
from datetime import datetime


LOG = f"{datetime.now().strftime('%d/%m/%Y')}.log"

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%S')}.log"

LOG_DIR = os.path.join(os.getcwd(),"logs",LOG)

if not os.path.isdir(LOG_DIR):
  os.makedirs(LOG_DIR)

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE)

logging.basicConfig(
  filename=LOG_FILE_PATH,
  format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
  level=logging.INFO
)



