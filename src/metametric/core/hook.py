from abc import ABC, abstractmethod
from typing import Dict, Generic, TypeVar, List, Iterable, Callable

T = TypeVar("T")




class Hook(ABC, Generic[T]):

    def __init__(self, callback: Callable[[T, T, str, str, float], None]):
        self.callback = callback

    def on_match(self, pred: T, ref: T, pred_path: str, ref_path: str, score: float):
        self.callback(pred, ref, pred_path, ref_path, score)

