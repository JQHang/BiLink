import gc
import sys
import time
import logging
from pyspark.sql import SparkSession

# Get logger
logger = logging.getLogger(__name__)

def start_spark_localhost(mode='local', config_dict=None, spark_log_level="ERROR"):
    """
    Initialize and return a SparkSession for localhost platform.

    This platform is designed for local debugging with small-scale data.
    It binds the Spark driver to 127.0.0.1 to avoid network issues.

    Args:
        mode (str): Execution mode. Only 'local' is supported for localhost platform.
        config_dict (dict, optional): Additional Spark configuration parameters.
        spark_log_level (str): Spark log level (default: "ERROR").

    Returns:
        SparkSession: The initialized Spark session.

    Raises:
        ValueError: If mode is not 'local'.
    """
    # Validate that only local mode is supported
    if mode != 'local':
        raise ValueError(
            f"localhost platform only supports 'local' mode, got '{mode}'. "
            "For cluster execution, use a different platform (e.g., 'example')."
        )

    # Initialize SparkSession builder
    builder = SparkSession.builder.appName("JoinMiner_Localhost_SparkSession")

    # Configure for local mode with localhost binding
    builder = builder.enableHiveSupport().master("local[*]")
    logger.info("Running in localhost platform (local mode for debugging)...")

    # Default configurations for localhost debugging
    default_configs = {
        # Bind to localhost to avoid network issues
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.driver.host": "127.0.0.1",

        # Memory and cores (adjust based on local machine)
        "spark.driver.memory": "4g",
        "spark.driver.cores": "2",
        "spark.driver.maxResultSize": "0",

        # Basic configurations for local debugging
        "spark.sql.shuffle.partitions": "8",  # Smaller for local testing
        "spark.default.parallelism": "8",
        "spark.sql.sources.partitionOverwriteMode": "dynamic",
    }

    # Merge user-provided configurations
    if config_dict:
        default_configs.update(config_dict)

    # Apply all configurations
    for key, value in default_configs.items():
        builder = builder.config(key, value)

    # Create Spark session
    spark = builder.getOrCreate()

    # Set log level
    spark.sparkContext.setLogLevel(spark_log_level)

    # Log application info
    app_id = spark.sparkContext.applicationId
    logger.info(f"Localhost Spark Application ID: {app_id}")
    logger.info(f"Spark UI: http://127.0.0.1:4040")

    return spark

def handle_error_localhost(spark_runner, error):
    """
    Handle Spark errors for localhost platform.

    On localhost, there is no external system that would terminate Spark.
    Any Spark error indicates a real problem (bug, configuration issue, resource exhaustion).
    Therefore, we fail immediately without attempting restart.

    Args:
        spark_runner: SparkRunner instance that encountered the error.
        error: The exception that was caught.

    Raises:
        The original exception (always).
    """
    logger.error(f"Unrecoverable Spark error on localhost platform: {type(error).__name__}: {error}")
    logger.info("Localhost does not support automatic restart - error indicates real problem")
    raise error
