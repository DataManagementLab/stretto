from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def get_operation_identifier(self) -> str:
        raise NotImplementedError
