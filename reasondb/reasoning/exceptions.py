from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from reasondb.query_plan.logical_plan import LogicalPlanStep
    from reasondb.query_plan.plan import PlanStep


class ReasoningDeadEnd(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


class Mistake(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


class ValidationError(Mistake):
    def __init__(self, step: "PlanStep", message: str):
        self.step = step
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message} in step {self.step}"


class ParsingError(Mistake):
    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"
