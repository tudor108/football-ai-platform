"""
Logging utilities for ML pipelines.
- Sets up standard logging format
"""
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
