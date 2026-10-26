import logging
import os
import sys
from typing import Optional

# Third-party loggers that are too chatty at INFO
QUIET_LOGGERS = (
    "botocore",
    "boto3",
    "urllib3",
    "s3transfer",
    "redshift_connector",
    "filelock",
    "httpx",
)


class CustomFormatter(logging.Formatter):
    """Custom formatter to modify pathname in logs by removing the current directory."""  # noqa: E501

    def format(self, record):
        cwd = os.getcwd()
        if record.pathname.startswith(cwd):
            record.pathname = record.pathname[len(cwd) + 1 :]
        return super(CustomFormatter, self).format(record)


def setup_logger(
    log_file: Optional[str] = None, level: int = logging.INFO
) -> None:
    """Configure the root logger.

    Console (stdout, what SageMaker sends to CloudWatch) at `level`.
    log_file: optional extra file with DEBUG logs; inside a container the
    file is lost after the job, so it is off by default.
    """
    logger = logging.getLogger()  # Configure the root logger
    logger.setLevel(logging.DEBUG if log_file else level)

    if logger.hasHandlers():
        # Clear existing handlers to avoid duplicate logs
        logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        CustomFormatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            CustomFormatter(
                "%(asctime)s:%(levelname)s:%(pathname)s:%(funcName)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)

    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
