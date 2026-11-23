import json
import logging
import sys
import time
from typing import Any, Callable, Optional

SPEAK_LEVEL = 25
logging.addLevelName(SPEAK_LEVEL, "SPEAK")


class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to console output."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[34;20m"
    cyan = "\x1b[36;20m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: f"{grey}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.INFO: f"{green}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        SPEAK_LEVEL: f"{cyan}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.WARNING: f"{yellow}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.ERROR: f"{red}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
        logging.CRITICAL: f"{bold_red}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset}",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class _CallableSink:
    def __init__(self, callback: Callable[[str, str], Any]):
        self.callback = callback

    def emit(self, level: str, message: str) -> None:
        self.callback(level, message)


class CustomLogger:
    def __init__(
        self,
        name: str = "AppLogger",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        console: bool = True,
        sinks: Optional[list[Any]] = None,
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        self._log_level = level
        self._sinks: list[Any] = []
        self._handlers_configured = False
        self._ui_publisher: Optional[Callable[[dict[str, Any]], Any]] = None
        self._speech_publisher: Optional[Callable[[str], Any]] = None
        self.speech_mode = 'current'

        if sinks:
            for sink in sinks:
                self.add_sink(sink)

        self.configure(console=console, log_file=log_file)

    def configure(
        self,
        *,
        console: bool = True,
        log_file: Optional[str] = None,
        ui_publisher: Optional[Callable[[dict[str, Any]], Any]] = None,
    ) -> None:
        self._reset_handlers()
        if console:
            c_handler = logging.StreamHandler(sys.stdout)
            c_handler.setFormatter(ColorFormatter())
            self.logger.addHandler(c_handler)

        if log_file:
            f_handler = logging.FileHandler(log_file)
            f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            f_handler.setFormatter(f_format)
            self.logger.addHandler(f_handler)

        if ui_publisher is not None:
            self.set_ui_publisher(ui_publisher)

    def add_sink(self, sink: Any) -> None:
        if callable(sink) and not hasattr(sink, "emit"):
            self._sinks.append(_CallableSink(sink))
        else:
            self._sinks.append(sink)

    def set_ui_publisher(self, publisher: Optional[Callable[[dict[str, Any]], Any]], log_level: int = logging.WARNING) -> None:
        self._ui_publisher = publisher
        if publisher is None:
            return

        if not any(getattr(sink, "__class__", None).__name__ == "UiPublisherSink" for sink in self._sinks):
            self._sinks.append(_UiPublisherSink(publisher, log_level))

    def set_speech_publisher(self, publisher: Optional[Callable[[str], Any]]) -> None:
        self._speech_publisher = publisher
        self._sinks = [sink for sink in self._sinks if not isinstance(sink, _SpeechPublisherSink)]
        if publisher is None:
            return
        self._sinks.append(_SpeechPublisherSink(
            lambda text: self._speech_publisher(text) if self.speech_mode != 'off' and self._speech_publisher else None))

    def set_speech_mode(self, mode: str) -> None:
        if mode not in {'current', 'off', 'tmc'}:
            raise ValueError('speech mode must be current, off or tmc')
        self.speech_mode = mode

    def reset(self) -> None:
        self._sinks = []
        self._speech_publisher = None
        self.speech_mode = 'current'
        self._reset_handlers()
        self.configure(console=True, log_file=None)

    def _reset_handlers(self) -> None:
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    def _log_print_style(self, level: int, msg, *args, exc_info=False):
        full_msg = self._format_message(msg, args)
        self.logger.log(level, full_msg, exc_info=exc_info)
        self._dispatch_to_sinks(level, full_msg)

    @staticmethod
    def _format_message(msg: Any, args: tuple[Any, ...]) -> str:
        text = str(msg)
        if not args:
            return text

        try:
            return text % args
        except (TypeError, ValueError):
            return " ".join([text, *[str(arg) for arg in args]])

    def _dispatch_to_sinks(self, level: int, message: str) -> None:
        if level < self._log_level:
            return
        level_name = self._level_to_name(level)
        for sink in list(self._sinks):
            try:
                if hasattr(sink, "emit"):
                    sink.emit(level_name, message)
                elif callable(sink):
                    sink(level_name, message)
            except Exception:  # noqa: BLE001
                self.logger.exception("Logger sink failed")

    @staticmethod
    def _level_to_name(level: int) -> str:
        if level == SPEAK_LEVEL:
            return "speak"
        if level >= logging.ERROR:
            return "error"
        if level >= logging.WARNING:
            return "warn"
        if level >= logging.INFO:
            return "info"
        return "debug"

    def debug(self, msg, *args):
        self._log_print_style(logging.DEBUG, msg, *args)

    def info(self, msg, *args):
        self._log_print_style(logging.INFO, msg, *args)

    def speak(self, msg, *args):
        self._log_print_style(SPEAK_LEVEL, msg, *args)

    def warning(self, msg, *args):
        self._log_print_style(logging.WARNING, msg, *args)

    def error(self, msg, *args):
        self._log_print_style(logging.ERROR, msg, *args)

    def critical(self, msg, *args):
        self._log_print_style(logging.CRITICAL, msg, *args)

    def exception(self, msg, *args):
        self._log_print_style(logging.ERROR, msg, *args, exc_info=True)


class _UiPublisherSink:
    def __init__(self, publisher: Callable[[dict[str, Any]], Any], log_level: int):
        self.publisher = publisher
        self._log_level = log_level
        self.disabled = False#

    @staticmethod
    def _name_to_level(level: str) -> int:
        if level == "speak":
            return SPEAK_LEVEL
        if level == "error":
            return logging.ERROR
        if level == "warn":
            return logging.WARNING
        if level == "info":
            return logging.INFO
        return logging.DEBUG

    def emit(self, level: str, message: str) -> None:
        if self.disabled or self._name_to_level(level) < self._log_level:
            return
        payload = {"level": level, "message": message, "timestamp": time.time()}
        try:
            self.publisher(payload)
        except Exception:  # noqa: BLE001
            self.disabled = True
            raise


class _SpeechPublisherSink:
    def __init__(self, publisher: Callable[[str], Any]):
        self.publisher = publisher
        self.disabled = False

    def emit(self, level: str, message: str) -> None:
        if self.disabled or level != "speak":
            return
        try:
            self.publisher(message)
        except Exception:  # noqa: BLE001
            self.disabled = True
            raise


# Global instance for easy access
log = CustomLogger()


def configure_logging(*, console: bool = True, log_file: Optional[str] = None, ui_publisher: Optional[Callable[[dict[str, Any]], Any]] = None) -> None:
    log.configure(console=console, log_file=log_file, ui_publisher=ui_publisher)


def set_ui_publisher(publisher: Optional[Callable[[dict[str, Any]], Any]]) -> None:
    log.set_ui_publisher(publisher)


def set_speech_publisher(publisher: Optional[Callable[[str], Any]]) -> None:
    log.set_speech_publisher(publisher)


def set_speech_mode(mode: str) -> None:
    log.set_speech_mode(mode)


def get_speech_mode() -> str:
    return log.speech_mode


def add_sink(sink: Any) -> None:
    log.add_sink(sink)


def reset() -> None:
    log.reset()


def debug(msg, *args):
    log.debug(msg, *args)


def info(msg, *args):
    log.info(msg, *args)


def speak(msg, *args):
    log.speak(msg, *args)


def warn(msg, *args):
    log.warning(msg, *args)


def warning(msg, *args):
    log.warning(msg, *args)


def error(msg, *args):
    log.error(msg, *args)


def critical(msg, *args):
    log.critical(msg, *args)


def exception(msg, *args):
    log.exception(msg, *args)
