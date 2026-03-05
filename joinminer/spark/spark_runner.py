import logging
from dataclasses import dataclass
from py4j.protocol import Py4JError, Py4JJavaError, Py4JNetworkError
from pyspark.sql import SparkSession

from joinminer.utils import time_costing
from joinminer.spark.managers.persist import PersistManager
from joinminer.spark.managers.table_state import TableStateManager
from joinminer.fileio import FileIO

# Get logger
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparkContext:
    """
    Immutable context object passed to task functions in SparkRunner.run().

    This dataclass encapsulates all the resources needed by task functions,
    providing a clear separation between the SparkRunner (orchestrator) and
    the task context (execution environment).

    Attributes:
        spark: Active SparkSession for DataFrame operations
        fileio: FileIO instance for file system operations
        persist_manager: PersistManager for DataFrame caching
        table_state: TableStateManager for table completion tracking
    """
    spark: SparkSession
    fileio: FileIO
    persist_manager: PersistManager
    table_state: TableStateManager

# Import platform-specific implementations
from joinminer.spark.platforms.example import start_spark_example, handle_error_example
from joinminer.spark.platforms.localhost import start_spark_localhost, handle_error_localhost

# Platform configuration registry
PLATFORM_CONFIG = {
    'example': {
        'start_spark': start_spark_example,
        'handle_error': handle_error_example
    },
    'localhost': {
        'start_spark': start_spark_localhost,
        'handle_error': handle_error_localhost
    }
    # Future platforms can be added here
    # 'aws': {
    #     'start_spark': start_spark_aws,
    #     'handle_error': handle_error_aws
    # }
}

class SparkRunner:
    """
    A Spark runner that handles errors and automatically restarts Spark sessions.

    This class provides a fault-tolerant way to run Spark jobs with automatic retry logic,
    platform-specific configurations, integrated DataFrame persistence management,
    and multi-filesystem support.

    Attributes:
        mode (str): Execution mode ('local' or 'cluster').
        config_dict (dict): Additional Spark configuration parameters.
        max_restarts (int): Maximum number of restart attempts per run() call.
        spark_log_level (str): Spark log level.
        platform (str): Platform identifier (e.g., 'example').
        spark: The active Spark session.
        persist_manager: PersistManager instance for managing DataFrame persistence.
        table_state: TableStateManager instance for managing table completion state.
        fileio: FileIO instance for file I/O operations.
    """

    def __init__(self, mode='cluster', platform=None, fileio=None,
                 config_dict=None, max_restarts=20, spark_log_level="ERROR",
                 ignore_complete=False):
        """
        Initialize the SparkRunner.

        Args:
            mode (str): Execution mode ('local' or 'cluster').
            platform (str): Platform identifier (required). Supported: 'example', 'localhost'.
            fileio (FileIO): FileIO instance (required). Used for table state management and file I/O operations.
            config_dict (dict, optional): Additional Spark configuration parameters.
            max_restarts (int): Maximum number of restart attempts (default: 20).
            spark_log_level (str): Spark log level (default: "ERROR").
            ignore_complete (bool): If True, all table completeness checks will be ignored,
                                   forcing rebuild of all tables. Useful for --force flag
                                   in scripts. (default: False)

        Raises:
            ValueError: If platform or fileio is not provided, or if platform is not supported.

        Note:
            The fileio parameter is accessible as a public attribute and can be used for
            file I/O operations such as reading/writing metadata files.
        """
        self.mode = mode
        self.config_dict = config_dict or {}
        self.max_restarts = max_restarts
        self.spark_log_level = spark_log_level

        # Validate platform is provided
        if platform is None:
            raise ValueError(
                "Platform parameter is required. Please specify a platform "
                f"from: {list(PLATFORM_CONFIG.keys())}"
            )

        self.platform = platform

        # Validate platform is supported
        if platform not in PLATFORM_CONFIG:
            raise ValueError(f"Unsupported platform: {platform}. "
                           f"Supported platforms: {list(PLATFORM_CONFIG.keys())}")

        # Validate fileio is provided
        if fileio is None:
            raise ValueError(
                "FileIO instance is required. Initialize FileIO before creating SparkRunner.\n"
                "Example: fileio = FileIO({'local': {}})"
            )

        self.fileio = fileio
        logger.info(f"SparkRunner using FileIO with schemes: {list(fileio.backends.keys())}")

        # Set up platform-specific methods
        platform_config = PLATFORM_CONFIG[platform]
        self._start_spark = lambda: platform_config['start_spark'](
            self.mode, self.config_dict, self.spark_log_level
        )
        self._handle_error = lambda error: platform_config['handle_error'](self, error)

        # Initialize Spark session
        logger.info(f"Initializing SparkRunner for platform '{platform}' "
                   f"in {mode} mode...")
        self.spark = self._start_spark()

        # Initialize PersistManager
        self.persist_manager = PersistManager()
        logger.info("PersistManager initialized")

        # Initialize TableStateManager
        self.table_state = TableStateManager(self.fileio, ignore_complete=ignore_complete)
        logger.info(f"TableStateManager initialized (ignore_complete={ignore_complete})")

    @time_costing
    def run(self, task_function, *args, **kwargs):
        """
        Execute a task function with automatic error handling and restart logic.

        This method will run the provided task function with a SparkContext.
        If a Spark-related error occurs, it will automatically handle the error
        and restart the Spark session according to platform-specific logic.

        Args:
            task_function: A function that takes a SparkContext instance (as spark_ctx) as its first argument.
            *args: Additional positional arguments to pass to the task function.
            **kwargs: Additional keyword arguments to pass to the task function.

        Returns:
            The result of the task function, or None if execution failed.

        Example:
            def process_data(spark_ctx: SparkContext, input_path, output_path):
                df = spark_ctx.spark.read.parquet(input_path)
                df_cached = spark_ctx.persist_manager.persist(df, save_point='output')
                # Process data...
                df.write.parquet(output_path)
                spark_ctx.persist_manager.mark_saved('output')

            runner = SparkRunner(mode='cluster', platform='localhost', fileio=fileio)
            result = runner.run(process_data, "hdfs://input", "hdfs://output")
        """
        result = None
        attempt_count = 0

        while attempt_count < self.max_restarts:
            attempt_count += 1
            try:
                logger.info(f"Executing task (attempt {attempt_count}/{self.max_restarts})...")

                # Create immutable context for the task function
                spark_ctx = SparkContext(
                    spark=self.spark,
                    fileio=self.fileio,
                    persist_manager=self.persist_manager,
                    table_state=self.table_state
                )

                result = task_function(spark_ctx, *args, **kwargs)
                logger.info("Task completed successfully")
                break

            except Exception as error:
                logger.warning(f"Error occurred during attempt {attempt_count}/{self.max_restarts}: "
                             f"{type(error).__name__}")

                # Check if we've exhausted retries
                if attempt_count >= self.max_restarts:
                    logger.error(f"Maximum restart attempts ({self.max_restarts}) exceeded")
                    raise

                # Handle error (may raise if unrecoverable)
                # Platform-specific logic determines if error is recoverable
                # If recoverable: restarts Spark and returns normally
                # If unrecoverable: re-raises the original exception
                self._handle_error(error)

                logger.info("Error handled successfully, retrying task...")

        return result

    def stop(self):
        """
        Manually stop the Spark session and release all persisted DataFrames.

        This method should be called when you're done using the runner
        to clean up resources.
        """
        # Release all persisted DataFrames first
        try:
            released_count = self.persist_manager.release_all()
            if released_count > 0:
                logger.info(f"Released {released_count} persisted DataFrames")
        except Exception as e:
            logger.error(f"Error releasing persisted DataFrames: {e}")

        # Stop Spark session
        if self.spark:
            try:
                self.spark.stop()
                logger.info("Spark session stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Spark session: {e}")
            finally:
                self.spark = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures Spark session is stopped."""
        self.stop()
        return False
