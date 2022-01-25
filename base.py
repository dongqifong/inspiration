
from typing import Optional


class Base:
    def __init__(self, func) -> None:
        self.func = func
        pass

    def fit(self, x, **kwargs):
        print(kwargs)
        x, y = self.func(x, **kwargs)
        return x, y
