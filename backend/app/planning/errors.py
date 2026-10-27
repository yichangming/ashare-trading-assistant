class PlanningError(Exception):
    """Base class for deterministic planning rejections."""


class EntryInvalidError(PlanningError):
    pass


class StopTooWideError(PlanningError):
    pass


class PositionRejectedError(PlanningError):
    pass


class PoorRiskRewardError(PlanningError):
    pass
