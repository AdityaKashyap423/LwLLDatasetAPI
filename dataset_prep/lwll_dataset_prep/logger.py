import logging
from typing import Any

class Logger():

    def __init__(self, level: str) -> None:
        logging.basicConfig()
        self.logger = logging.getLogger("logger_main")

        if level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise Exception("Invalid logging level")

        if level == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif level == "INFO":
            self.logger.setLevel(logging.INFO)
        elif level == "WARNING":
            self.logger.setLevel(logging.WARNING)
        elif level == "ERROR":
            self.logger.setLevel(logging.ERROR)

    def debug(self, msg: Any) -> None:
        self.logger.debug(msg)

    def info(self, msg: Any) -> None:
        self.logger.info(msg)

    def warning(self, msg: Any) -> None:
        self.logger.warning(msg)

    def error(self, msg: Any) -> None:
        self.logger.error(msg)


log = Logger('INFO')
