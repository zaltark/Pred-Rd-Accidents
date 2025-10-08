# Logging Strategy

This document outlines the logging strategy for the project.

## Configuration

The logging configuration is defined in the `LOGGING_CONFIG` dictionary in `config/config.py`. This centralized configuration allows us to easily manage logging behavior across the entire application.

## Log Levels

We will use the standard Python log levels to categorize messages:

*   **DEBUG**: Detailed information, typically of interest only when diagnosing problems.
*   **INFO**: Confirmation that things are working as expected.
*   **WARNING**: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
*   **ERROR**: Due to a more serious problem, the software has not been able to perform some function.
*   **CRITICAL**: A serious error, indicating that the program itself may be unable to continue running.

## Log Output

Our logging setup includes two handlers:

1.  **Console Handler**: This handler streams log records to the console. It is configured to display `INFO` level logs and above, providing real-time feedback during development and execution.

2.  **File Handler**: This handler writes log records to a rotating file located at `logs/app.log`. It is configured to capture `DEBUG` level logs and above, providing a persistent and detailed record of the application's execution. The log file will rotate when it reaches 10MB, and up to 5 backup files will be kept.

## Usage

To use the logger in any module, you can add the following code:

```python
import logging
import logging.config
from config.config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

logger.info("This is an info message.")
logger.debug("This is a debug message.")
```
