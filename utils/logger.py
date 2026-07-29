"""
==================================================
Logger Utility
==================================================

This module provides a centralized logger for the
Advanced PDF RAG project.

Every module should import this logger instead of
using print() statements.

Example
-------
from utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Loading document...")
==================================================
"""

# --------------------------------------------------
# Imports
# --------------------------------------------------

import logging
import sys

from config import LOG_LEVEL


# --------------------------------------------------
# Logger Format
# --------------------------------------------------

LOG_FORMAT = (
    "[%(asctime)s] "
    "[%(levelname)s] "
    "[%(name)s] "
    "%(message)s"
)

DATE_FORMAT = "%H:%M:%S"


# --------------------------------------------------
# Configure Root Logger
# --------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stdout,
)


# --------------------------------------------------
# Logger Factory
# --------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a logger for the given module.

    Parameters
    ----------
    name : str
        Usually __name__ from the calling module.

    Returns
    -------
    logging.Logger
    """

    return logging.getLogger(name)