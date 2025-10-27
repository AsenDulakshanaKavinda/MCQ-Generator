import os
import logging
from datetime import datetime
import structlog

class CustomLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.log_dir, log_file)

        self._configure_standard_logging()

    def _configure_standard_logging(self):
        """Set up base Python logging with file + console handlers."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(self.log_file_path),
                logging.StreamHandler()
            ]
        )

    def get_logger(self, name: str = __file__):
        logger_name = os.path.basename(name)

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)
