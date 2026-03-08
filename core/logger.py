import logging

from core.config import DEBUG_MODE


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger pre-configured with the [BMO] prefix format.

    All loggers share the same handler format: "[BMO] message".
    Log level is controlled globally by DEBUG_MODE in core/config.py:
        - False (default) → INFO and above only
        - True            → DEBUG and above (verbose internal state)
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[BMO] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Prevent log records from propagating to the root logger,
        # which would double-print if root has its own handler.
        logger.propagate = False

    logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
    return logger
