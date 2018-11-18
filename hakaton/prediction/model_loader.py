from abc import ABC, abstractmethod

from hakaton.prediction.model import SkyhacksModel


class ModelLoader(ABC):

    @abstractmethod
    def load(self) -> SkyhacksModel:
        pass
