import logging
from datetime import datetime
from functools import wraps

from .logger import setup_logger

# 获得logger
logger = logging.getLogger(__name__)

def time_costing(func):
    """
    A decorator that logs or prints the start time, end time, and duration of the execution
    of a function. It checks if a 'logger' key is provided in keyword arguments and if it
    is a valid logger instance. If a valid logger is provided, it uses it for logging.
    Otherwise, it uses print for output.

    Parameters:
    func (callable): The function to be decorated.

    Returns:
    callable: The wrapped function that now logs or prints its execution details.
    """
    def wrapper(*args, **kwargs):
        # Log or print the beginning of the operation
        start = datetime.now()
        logger.info(f"Start Function: {func.__name__}")
        
        # Execute the decorated function
        result = func(*args, **kwargs)
        
        # Log or print the end of the operation
        end = datetime.now()
        duration = end - start
        logger.info(f"End Function: {func.__name__}, Time Costing: {duration}")
        
        return result
    return wrapper
