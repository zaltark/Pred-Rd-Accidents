# config/config.py

# Data paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"

# Model parameters
MODEL_NAME = "random_forest"
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "logs/app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"],
    },
}
