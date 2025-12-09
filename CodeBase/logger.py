#!/usr/bin/env python3
"""
Unified logger for the entire fact-checking pipeline.

Goals:
    - Clean, readable logs
    - Consistent formatting across all modules
    - Safe for multi-pass + multi-stage pipeline
    - Writes to both console + log file
"""

import logging
from pathlib import Path

# Log file location
LOG_DIR = Path("output")
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_PATH = LOG_DIR / "pipeline.log"


# ---------------------------------------------------------------
# Create global logger
# ---------------------------------------------------------------
logger = logging.getLogger("fact_checker")
logger.setLevel(logging.INFO)     # default log level


# ---------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ---------------------------------------------------------------
# Console Handler
# ---------------------------------------------------------------
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)


# ---------------------------------------------------------------
# File Handler
# ---------------------------------------------------------------
file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)


# ---------------------------------------------------------------
# Attach handlers (avoid duplicates)
# ---------------------------------------------------------------
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
