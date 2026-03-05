import logging
import os

def setup_logger(log_file=None, level='DEBUG'):
    """
    Set up and return the joinminer package logger with handlers configured for
    console output and optionally file output. The logger name is automatically
    determined from the package name.

    It also ensures no duplicate handlers are added.

    Args:
        log_file (str, optional): Full path to the log file (e.g., '/path/to/logs/task/20250113_143022.log').
                                  If None, only logs to console without file output.
        level (str): The log level as a string (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        logging.Logger: A configured logger that logs to console and optionally to the specified file.

    Example:
        >>> from datetime import datetime
        >>> timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        >>> log_file = f"/path/to/logs/task_name/{timestamp}.log"
        >>> logger = setup_logger(log_file, level='INFO')
    """
    # Convert the level string to a logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Automatically get package name from module name
    # __name__ = "joinminer.utils.logger" -> logger_name = "joinminer"
    logger_name = __name__.split('.')[0]

    # Get or create the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    logger.propagate = False

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create and configure a handler for outputting to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log_file path is provided, add a file handler
    if log_file:
        # Create parent directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Messages stored at {os.path.abspath(log_file)}")

    return logger