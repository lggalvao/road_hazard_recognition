import functools
import time
import logging
from collections import defaultdict

# Global storage for timings
_execution_times = defaultdict(list)

def timeit(fn):
    """
    Decorator to log execution time and store all durations.
    At the end, you can compute average execution time per function.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Store the timing
        _execution_times[fn.__name__].append(elapsed)

        #logging.info(f"[TIMER] {fn.__name__} took {elapsed:.4f} seconds")
        return result

    return wrapper

def print_average_timings():
    """
    Prints the average execution time for all functions decorated with @timeit.
    """
    logging.info("---- Average execution times ----")
    for fn_name, times in _execution_times.items():
        avg_time = sum(times) / len(times)
        logging.info(f"{fn_name}: {avg_time:.4f} s over {len(times)} calls")



def timeit_gpu(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        result = fn(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        logging.info(f"[TIMER] {fn.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper
