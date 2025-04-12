import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")
# The above code sets up logging in Python with a specific configuration.
# It includes a file handler to log messages to 'app.log' and a stream handler to log messages to the console.
# The logging level is set to DEBUG, which means all messages at this level and above will be logged.
# The format of the log messages includes the timestamp, logger name, log level, and the actual message.
# The date format is also specified. The logger is then used to log messages at different severity levels.
# The messages will be written to both the log file and the console.
# The logging module is a standard Python module used for logging messages in applications.
# It provides a flexible framework for emitting log messages from Python programs.
# The logging module is part of the Python standard library and is used to log messages from Python applications.
# The logging module is a standard Python module used for logging messages in applications.
# It provides a flexible framework for emitting log messages from Python programs.
# The logging module is part of the Python standard library and is used to log messages from Python applications.
# The logging module is a standard Python module used for logging messages in applications.
# It provides a flexible framework for emitting log messages from Python programs.
# The logging module is part of the Python standard library and is used to log messages from Python applications.
# The logging module is a standard Python module used for logging messages in applications.
# It provides a flexible framework for emitting log messages from Python programs.
# The logging module is part of the Python standard library and is used to log messages from Python applications.
# The logging module is a standard Python module used for logging messages in applications.
# It provides a flexible framework for emitting log messages from Python programs.
