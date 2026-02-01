from typing import Union
import abc


class TuningParameter(abc.ABC):
    def __init__(
        self,
        name: str,
        default: Union[int, float, str],
        init: Union[int, float, str],
    ):
        self._name = name
        self._default = default
        self._init = init

    @property
    def name(self) -> str:
        return self._name

    @property
    @abc.abstractmethod
    def default(self) -> Union[int, float, str]:
        """Default value of the tuning parameter"""
        pass

    @property
    @abc.abstractmethod
    def init(self) -> Union[int, float, str]:
        """Default value of the tuning parameter"""
        pass


class TuningParameterContinuous(TuningParameter):
    def __init__(
        self,
        name: str,
        default: float,
        init: float,
        min: float,
        max: float,
        log_scale: bool,
    ):
        super().__init__(
            name=name,
            default=default,
            init=init,
        )
        self.min = min
        self.max = max
        self.log_scale = log_scale

    @property
    def default(self) -> float:
        assert isinstance(self._default, float)
        return self._default

    @property
    def init(self) -> float:
        assert isinstance(self._init, float)
        return self._init


class TuningParameterDiscrete(TuningParameter):
    def __init__(
        self,
        name: str,
        default: int,
        init: int,
        min: int,
        max: int,
        log_scale: bool,
    ):
        super().__init__(
            name=name,
            default=default,
            init=init,
        )
        self.min = min
        self.max = max
        self.log_scale = log_scale

    @property
    def default(self) -> int:
        assert isinstance(self._default, int)
        return self._default

    @property
    def init(self) -> float:
        assert isinstance(self._init, int)
        return self._init


class TuningParameterCategorical(TuningParameter):
    def __init__(
        self,
        name: str,
        default: str,
        init: str,
        choices: list[str],
    ):
        super().__init__(
            name=name,
            default=default,
            init=init,
        )
        self.choices = choices

    @property
    def default(self) -> str:
        assert isinstance(self._default, str)
        return self._default

    @property
    def init(self) -> float:
        assert isinstance(self._init, str)
        return self._init
