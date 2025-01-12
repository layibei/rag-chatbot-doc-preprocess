import sys
from loguru import logger

# Initialize basic logger configuration
logger.remove()  # Remove default handler
logger.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level.icon} {level.name:<8} | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
    level="DEBUG"
) 