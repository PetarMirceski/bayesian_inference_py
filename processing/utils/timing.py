from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timing(title: str = "") -> Iterator[None]:
    start = perf_counter()
    yield
    ellapsed_time = perf_counter() - start
    print(title, ellapsed_time, " s")
