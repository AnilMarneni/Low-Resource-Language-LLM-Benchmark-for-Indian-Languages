import sys
from loguru import logger
import os

def setup_logger(log_file="benchmark.log"):
    """
    Configure loguru logger for the project.
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    os.makedirs("logs", exist_ok=True)
    logger.add(f"logs/{log_file}", rotation="10 MB", level="DEBUG")
    
    return logger

log = setup_logger()
