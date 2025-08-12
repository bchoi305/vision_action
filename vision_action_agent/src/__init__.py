from loguru import logger
import os
import sys

# Configure Loguru to always print to console; optional file sink via env
logger.remove()
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

# Console sink (no enqueue to avoid env restrictions)
logger.add(
    sys.stderr,
    level=log_level,
    colorize=True,
    enqueue=False,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"
)

# Optional file sink
log_file = os.environ.get("LOG_FILE_PATH")
if log_file:
    logger.add(
        log_file,
        level=log_level,
        rotation=os.environ.get("LOG_ROTATION", "10 MB"),
        retention=os.environ.get("LOG_RETENTION", "5"),
        enqueue=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )

logger.info("Logging configured (console + optional file)")

from .core.agent import VisionActionAgent
