"""
Structured logging helpers for this project.

This module does NOT own the logger. The project configures the *root* logger in
`BaseEngine.configure_persistence()` (per-rank file handler + rank-0 console
handler). All modules grab it with `logging.getLogger()`. To keep that design
intact while enriching every log line, this module provides two add-ons that are
called from `configure_persistence()`:

- `install_structured_record_factory()`: wraps the log-record factory so every
  record (from any logger, including the root logger and third-party libraries)
  gains extra fields -- most importantly `classname` / `qualname`
  (`ClassName.method`), which a standard LogRecord cannot provide. The format
  string then references them, e.g. `%(qualname)s`.
- `install_excepthooks(logger)`: routes *uncaught* exceptions (main thread,
  worker threads, and "unraisable" exceptions from `__del__` / weakref
  callbacks) into `logger`, so a crash lands in the per-rank log file instead of
  only on stderr (which a job launcher / redirected / detached process loses).

Adapted from the team `python-logging` skill. Kept compatible with `%`-style
formatting and the existing root-logger setup, so it composes cleanly with the
project's current logging rather than replacing it.
"""
import faulthandler
import logging
import os
import sys
import threading
import traceback


# `logging._srcfile` is the absolute path of the stdlib logging module's source
# file; the stdlib uses it to skip its own frames when locating a log call's
# origin. `_THIS_SRCFILE` is this module's path. Together they let
# `_caller_class_name` skip all the "plumbing" frames (stdlib logging + this
# module) sitting between a user's `logger.info(...)` and the record factory, so
# it can find the real call site. This means the actual callers must live in
# OTHER files than this one -- which every project module does.
_LOGGING_SRCFILE = logging._srcfile
_THIS_SRCFILE = os.path.normcase(__file__)


# Guard so repeated calls do not stack multiple wrapped factories on top of each
# other (each call would otherwise add another layer of indirection per record).
_record_factory_installed = False


def _current_process_rank() -> int:
    """Return this process's global rank, falling back to env vars then 0."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))


def _caller_class_name() -> str:
    """
    Return the class name of the function that issued the current log call.

    A standard LogRecord knows the function name (`funcName`) but not its class.
    The only way to recover the class is to inspect the live call stack: walk
    outward from the current frame, skipping every frame belonging to the stdlib
    `logging` module or to this module (the plumbing between `logger.info(...)`
    and the record factory). The first remaining frame is the real call site.

    There, the *static* tuple `f_code.co_varnames` is checked first (cheap): by
    convention the first local of an instance method is `self` and of a
    classmethod is `cls`. Only then is `f_locals` touched (which forces CPython
    to materialize the locals dict). Returns `""` for plain functions,
    staticmethods, and module-level code.
    """
    frame = sys._getframe()
    while frame is not None:
        co_file = frame.f_code.co_filename
        if co_file != _LOGGING_SRCFILE and co_file != _THIS_SRCFILE:
            varnames = frame.f_code.co_varnames
            if varnames:
                first = varnames[0]
                if first == "self":
                    return type(frame.f_locals["self"]).__name__
                if first == "cls":
                    return frame.f_locals["cls"].__name__
            return ""
        frame = frame.f_back
    return ""


def install_structured_record_factory() -> None:
    """
    Wrap the log-record factory so every LogRecord gains extra fields.

    `logging` builds each record via the factory from `logging.getLogRecordFactory()`.
    Wrapping (not subclassing) composes with any other library that also
    customized it: the existing factory builds the standard record first, then we
    decorate it. Fields added:

    - `rank`      : process rank (see `_current_process_rank`).
    - `relpath`   : source path relative to the cwd captured at install time
                    (stable across later `chdir`). Falls back to the bare file
                    name if no relative path exists.
    - `classname` : caller's class name (`""` for plain functions).
    - `qualname`  : `"ClassName.funcName"` when a class is known, else `funcName`.

    Idempotent: guarded so repeated calls install the wrapper only once.
    """
    global _record_factory_installed
    if _record_factory_installed:
        return

    factory = logging.getLogRecordFactory()
    cwd = os.getcwd()

    def record_factory(*args, **kwargs):
        record = factory(*args, **kwargs)
        record.rank = _current_process_rank()
        try:
            record.relpath = os.path.relpath(record.pathname, cwd)
        except ValueError:
            # Different drive on Windows -> no relative path exists.
            record.relpath = record.filename
        class_name = _caller_class_name()
        record.classname = class_name
        record.qualname = (
            f"{class_name}.{record.funcName}" if class_name else record.funcName
        )
        return record

    logging.setLogRecordFactory(record_factory)
    _record_factory_installed = True


def _flush_logger(target_logger: logging.Logger) -> None:
    """Best-effort flush of every handler so the last record reaches disk.

    During an interpreter teardown / dying-CUDA crash, buffered records can be
    lost if the process dies before handlers flush. We flush explicitly right
    after logging an uncaught exception.
    """
    for handler in target_logger.handlers:
        try:
            handler.flush()
        except Exception:
            pass


def _format_exc(exc_type, exc_value, exc_tb) -> str:
    """Render a traceback to a plain string.

    We embed the traceback directly into the log *message* instead of relying on
    the formatter's `exc_info` path. In a multi-process run with a dying CUDA
    context, the formatter's cached `exc_text` rendering was observed to come out
    empty; formatting it ourselves here is robust to that.
    """
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


def install_excepthooks(target_logger: logging.Logger) -> None:
    """
    Route *uncaught* exceptions through `target_logger`.

    Three hooks, because Python reports uncaught exceptions through three
    channels:

    1. `sys.excepthook` -- main thread. `KeyboardInterrupt` is passed to the
       default hook so Ctrl-C keeps normal behavior and is not logged.
    2. `threading.excepthook` -- worker threads. `SystemExit` is ignored (a
       thread exiting normally, not an error).
    3. `sys.unraisablehook` -- exceptions that cannot propagate, e.g. raised in a
       `__del__` finalizer or weakref callback.

    The full traceback is formatted into the log *message* (not passed via
    `exc_info`), because the formatter's `exc_info` rendering was observed to come
    out empty during a multi-process crash with a dying CUDA context. After
    logging, all handlers are flushed so the record reaches disk before the
    process dies.

    `faulthandler` is also enabled so a C-level fault (segfault, CUDA illegal
    access, abort) or a hang still dumps Python stacks of every thread to stderr.
    """
    # Dump Python tracebacks of all threads on a fatal C-level signal
    # (segfault / abort / bus error). Goes to the real stderr fd, which survives
    # even when the Python-level logging stack is wedged.
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass

    def _log_uncaught(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        target_logger.critical(
            "Uncaught exception:\n%s", _format_exc(exc_type, exc_value, exc_tb)
        )
        _flush_logger(target_logger)

    sys.excepthook = _log_uncaught

    def _log_thread_exc(args):
        if issubclass(args.exc_type, SystemExit):
            return
        target_logger.critical(
            "Uncaught exception in thread %s:\n%s",
            args.thread.name,
            _format_exc(args.exc_type, args.exc_value, args.exc_traceback),
        )
        _flush_logger(target_logger)

    threading.excepthook = _log_thread_exc

    def _log_unraisable(unraisable):
        target_logger.error(
            "Unraisable exception: %s\n%s",
            unraisable.err_msg or "",
            _format_exc(
                unraisable.exc_type,
                unraisable.exc_value,
                unraisable.exc_traceback,
            ),
        )
        _flush_logger(target_logger)

    sys.unraisablehook = _log_unraisable
