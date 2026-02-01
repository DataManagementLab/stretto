import os
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Optional


NOW = datetime.now()
FILE_HANDLER_LOGGERS: List[Optional["FileLogger"]] = [None] * 100
FILE_HANDLER_POINTER = 0


class FileLogger:
    def __init__(self, log_root_path: Optional[Path] = None):
        if log_root_path is None:
            log_root_path = Path("logging") / NOW.strftime("%Y-%m-%d--%H-%M-%S")
            log_root_path.mkdir(exist_ok=True, parents=True)
            if (Path("logging") / "latest").exists():
                os.unlink(Path("logging") / "latest")
            os.symlink(log_root_path.name, Path("logging") / "latest")

        self.log_root_path = log_root_path
        self.logger_name = ".".join(
            log_root_path.relative_to(Path("logging")).parts[1:]
        )
        self.file_handler: Optional[logging.FileHandler] = None

    def get_loggers(self, name: str):
        self.log_root_path.mkdir(exist_ok=True, parents=True)
        logger_name = name + ".~." + self.logger_name
        logger = logging.getLogger("file." + logger_name)
        logger.setLevel(logging.DEBUG)
        if len(logger.handlers) == 0:
            logger.addHandler(self.get_file_handler())
        else:
            self._file_handler = logger.handlers[0]
        logger.propagate = False
        return [logger, logging.getLogger(logger_name)]

    def get_file_handler(self):
        global FILE_HANDLER_POINTER

        if self.file_handler is None:
            close_logger = FILE_HANDLER_LOGGERS[FILE_HANDLER_POINTER]
            if close_logger is not None:
                assert close_logger.file_handler is not None
                close_logger.file_handler.close()
                close_logger.file_handler = None
            FILE_HANDLER_LOGGERS[FILE_HANDLER_POINTER] = self
            FILE_HANDLER_POINTER = (FILE_HANDLER_POINTER + 1) % len(
                FILE_HANDLER_LOGGERS
            )

            self.file_handler = logging.FileHandler(self.log_root_path / "logging.log")
            self.file_handler.setLevel(logging.DEBUG)
            self.file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        return self.file_handler

    def debug(self, key: str, first_message: str, *messages: str):
        for message in (first_message,) + messages:
            for logger in self.get_loggers(key):
                logger.debug(message)

    def info(self, key: str, first_message: str, *messages: str):
        for message in (first_message,) + messages:
            for logger in self.get_loggers(key):
                logger.info(message)

    def warning(self, key: str, first_message: str, *messages: str, exc_info=False):
        for message in (first_message,) + messages:
            for logger in self.get_loggers(key):
                logger.warning(message, exc_info=exc_info)

    def __truediv__(self, key: str) -> "FileLogger":
        return FileLogger(self.log_root_path / key)

    def error(self, key: str, first_message: str, *messages: str):
        for message in (first_message,) + messages:
            for logger in self.get_loggers(key):
                logger.error(message)


class NoLogger(FileLogger):
    def __init__(self):
        pass

    def get_loggers(self, name: str):
        logger = logging.getLogger(name + ".~." + self.logger_name)
        return [logger]
