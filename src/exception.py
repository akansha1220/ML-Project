import sys
import logging
import os
from src.logger import logging


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name= exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script [{0}] line number [{1}] error message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    
    return error_message



class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message


if __name__=="__main__":                       # "__name__" ue to check current directory name (yani kon is directory to call krra h)

    try:
        a=1/0
    except Exception as e:
        logging.info("dividing by 0",exc_info=True)
        logging.error("Throwing an error",exc_info=True)
        raise CustomException(e,sys)
