import time
import logging
from datetime import datetime
from pyspark.sql import SparkSession

# Get logger
logger = logging.getLogger(__name__)

def start_spark_example(mode='local', config_dict=None, spark_log_level="ERROR"):
    """
    Initialize and return a SparkSession for Example platform.
    
    For cluster mode, this function will wait until 9:00 AM if called before that time,
    as Example platform only allows cluster jobs during 9:00-24:00.
    
    Args:
        mode (str): Execution mode ('local' or 'cluster').
        config_dict (dict, optional): Additional Spark configuration parameters.
        spark_log_level (str): Spark log level (default: "ERROR").
        
    Returns:
        SparkSession: The initialized Spark session.
    """
    # Initialize SparkSession builder
    builder = SparkSession.builder.appName("Join_Miner_SparkSession")
    
    # Configure based on execution mode
    if mode == 'local':
        builder = builder.enableHiveSupport().master("local[*]")
        logger.info("Running in local mode...")
        default_configs = {
            "spark.driver.memory": "60g",
            "spark.driver.cores": "4",
            "spark.driver.maxResultSize": "0",
            "spark.sql.sources.partitionOverwriteMode": "dynamic"
        }
    else:
        # Wait until 9:00 AM for cluster mode
        current_time = datetime.now()
        if current_time.hour < 9:
            wait_until = datetime(current_time.year, current_time.month, 
                                current_time.day, 9, 0)
            wait_seconds = (wait_until - current_time).total_seconds()
            logger.info(f"Cluster mode: waiting until 9:00 AM to start. "
                       f"Waiting for {wait_seconds:.0f} seconds...")
            time.sleep(wait_seconds)
        
        builder = builder.enableHiveSupport()
        logger.info("Running in cluster mode...")
        default_configs = {
            "spark.archives": "hdfs:///user/example/bilink/pyspark_env.tar.gz#pyenv",
            
            # Basic Spark configurations
            "spark.default.parallelism": "1200",
            "spark.sql.shuffle.partitions": "2400",
            "spark.sql.broadcastTimeout": "3600",
            
            # Driver configurations
            "spark.driver.memory": "40g",
            "spark.driver.cores": "4",
            "spark.driver.maxResultSize": "0",
            
            # Executor configurations
            "spark.executor.memory": "32g",
            "spark.executor.cores": "4",
            "spark.dynamicAllocation.maxExecutors": "300",
            
            # SQL configurations
            "spark.sql.sources.partitionOverwriteMode": "dynamic",
            "spark.driver.allowMultipleContexts": "false",
            "spark.sql.autoBroadcastJoinThreshold": 100 * 1024 * 1024,  # 100MB
            "spark.sql.files.maxPartitionBytes": 256 * 1024 * 1024,     # 256MB

            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
            
            # Adaptive query execution
            "spark.sql.adaptive.enabled": "true",
            "spark.sql.adaptive.advisoryPartitionSizeInBytes": "128m",
            "spark.sql.adaptive.coalescePartitions.enabled": "true",
            "spark.sql.adaptive.skewJoin.enabled": "true",
            "spark.sql.adaptive.skewJoin.skewedPartitionFactor": "3",
            "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "256m"
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

    # Log Example platform specific URLs
    app_id = spark.sparkContext.applicationId
    logger.info(f"Application ID: http://example.cluster/proxy/{app_id}")
    logger.info(f"History Record: http://example.cluster/history/{app_id}/stages/")

    return spark


def handle_error_example(spark_runner, error):
    """
    Handle Spark errors for Example platform with automatic restart logic.

    Only restarts when detecting system-initiated shutdown (SparkContext was shut down).
    All other errors are unrecoverable and will be re-raised immediately.

    For system shutdowns:
    1. Skip cleanup (Spark already dead, stop() would fail)
    2. Wait until 9:00 AM if outside active hours (9:00-24:00)
    3. Restart Spark session
    4. Return normally to retry the task

    Args:
        spark_runner: SparkRunner instance that encountered the error.
        error: The exception that was caught.

    Raises:
        The original exception if not a system shutdown or if restart fails.
    """
    error_msg = str(error)

    # Check if this is a system-initiated shutdown
    if "SparkContext was shut down" not in error_msg:
        # Not a system shutdown - unrecoverable error
        logger.error(f"Unrecoverable Spark error on Example platform: {type(error).__name__}: {error}")
        logger.info("Error is not a system shutdown - failing immediately")
        raise error

    # System shutdown detected - attempt restart
    logger.info(f"Detected system shutdown: {error}")
    interrupt_time = datetime.now()
    logger.info(f"Spark shut down by system at: {interrupt_time}")

    # Stop old session to clear JVM SparkContext singleton
    # This prevents "only one SparkContext should be running" error
    try:
        spark_runner.spark.stop()
        logger.debug("Stopped old Spark session to clear JVM reference")
    except Exception as stop_error:
        logger.debug(f"Could not stop old session: {stop_error}")

    # Clear internal tracking (DataFrames are invalid after shutdown)
    spark_runner.persist_manager.clear_tracking()

    # Wait for JVM SparkContext cleanup (JVM cleanup is async, no API to detect)
    time.sleep(10)
    logger.debug("Waited 10s for JVM SparkContext cleanup")

    # Check if current time is within Example platform active hours (9:00-24:00)
    if interrupt_time.hour < 9:
        logger.info("Example platform: Operations not allowed outside 9:00-24:00, "
                   "waiting until next 9:00 AM...")

        # Calculate next 9:00 AM
        next_start_time = datetime(interrupt_time.year, interrupt_time.month,
                                 interrupt_time.day, 9, 0)

        wait_seconds = (next_start_time - interrupt_time).total_seconds()
        logger.info(f"Waiting for {wait_seconds:.0f} seconds until {next_start_time}...")
        time.sleep(wait_seconds)
        logger.info("Active hours have begun, attempting to restart...")

    # Attempt to restart Spark with retry logic (JVM may still be cleaning up)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            spark_runner.spark = spark_runner._start_spark()
            break
        except Exception as e:
            if "Only one SparkContext should be running" in str(e):
                if attempt < max_retries - 1:
                    logger.warning(f"SparkContext still active, retrying in 10s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(10)
                else:
                    raise RuntimeError(f"Failed to restart SparkContext after {max_retries} attempts") from e
            else:
                raise  # Other errors, fail immediately
    logger.info("Successfully restarted Spark session after system shutdown")
