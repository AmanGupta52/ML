import threading
from typing import Callable, Any, Optional
import functools
import queue


def threaded(fn: Callable) -> Callable:
    """
    Decorator to run a function in a separate daemon thread.
    Returns a ThreadResult object that allows checking status/results.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> 'ThreadResult':
        result_queue = queue.Queue()

        def target():
            try:
                res = fn(*args, **kwargs)
                result_queue.put(("success", res, None))
            except Exception as e:
                result_queue.put(("error", None, e))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        return ThreadResult(thread, result_queue)

    return wrapper


class ThreadResult:
    """
    Returned by @threaded functions.
    Allows:
    - checking if thread finished
    - retrieving result safely
    """
    def __init__(self, thread: threading.Thread, result_queue: queue.Queue):
        self.thread = thread
        self._queue = result_queue

    def ready(self) -> bool:
        """True if the thread has finished."""
        return not self.thread.is_alive()

    def result(self, timeout: Optional[float] = None) -> Any:
        """
        Get result (blocking until finished or timeout).
        Raises exception if wrapped function failed.
        """
        if not self.ready():
            self.thread.join(timeout)

        if self.thread.is_alive():
            raise TimeoutError("Thread did not complete in time")

        status, result, exc = self._queue.get_nowait()
        if status == "error":
            raise exc
        return result

    def result_or_none(self) -> Optional[Any]:
        """Non-blocking: return result if ready, else None."""
        if self.ready():
            try:
                return self.result(timeout=0)
            except:
                return None
        return None
