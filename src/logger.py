import logging
import os
import sys
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"            #creating a file name with current date and .log extention'''
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)                     #join the current working directory path logs folder filename '''
os.makedirs(logs_path,exist_ok= True)                                #even if existing file exist , keep making new one'''

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)                   #file path "log_path+log_file" '''

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] -%(lineno)s- %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)


if __name__=="__main__":
    logger = logging.getLogger()
    logging.info("Logging has started")