"""Machine-readable failure outcomes shared by actions and motion backends."""
from dataclasses import dataclass
from enum import Enum


class FailureCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    INVALID_STATE = "invalid_state"
    LIMIT_VIOLATION = "limit_violation"
    PLANNING_FAILED = "planning_failed"
    CONTROLLER_FAILED = "controller_failed"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    CANCELLED = "cancelled"
    CONDITION_FAILED = "condition_failed"
    UNKNOWN = "unknown"


class FailureOrigin(str, Enum):
    CONTROLLER = "controller"
    PERCEPTION = "perception"
    NETWORK = "network"
    INPUT = "input"


@dataclass(frozen=True)
class FailureInfo:
    code: FailureCode = FailureCode.UNKNOWN
    retryable: bool = False
    backend_code: int | str | None = None
    origin: FailureOrigin | None = None


def moveit_failure(error_code: int) -> FailureInfo | None:
    """MoveIt numeric outcomes; undocumented outcomes are never replayed."""
    if error_code == 1:
        return None
    if error_code in {-1, -2, -6, -31}:
        return FailureInfo(FailureCode.PLANNING_FAILED, retryable=True, backend_code=error_code)
    if error_code == -4:
        return FailureInfo(FailureCode.CONTROLLER_FAILED, backend_code=error_code)
    if error_code == -7:
        return FailureInfo(FailureCode.CANCELLED, backend_code=error_code)
    if error_code in {-3, -5, -26, -27, -28, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -21, -22, -23, -24}:
        return FailureInfo(FailureCode.INVALID_STATE, backend_code=error_code)
    return FailureInfo(backend_code=error_code)


class MotionFailure(RuntimeError):
    """A validation outcome with classification independent of its diagnostic text."""
    def __init__(self, reason: str, failure: FailureInfo):
        super().__init__(reason)
        self.failure = failure


def trajectory_failure(error_code: int | None, status: int | None) -> FailureInfo:
    """FollowJointTrajectory result codes and ROS action cancellation status."""
    if status == 5:  # action_msgs/GoalStatus.STATUS_CANCELED
        return FailureInfo(FailureCode.CANCELLED, backend_code=error_code)
    if error_code in {-1, -2, -3}:
        return FailureInfo(FailureCode.INVALID_REQUEST, backend_code=error_code)
    if error_code in {-4, -5}:
        return FailureInfo(FailureCode.LIMIT_VIOLATION, backend_code=error_code)
    return FailureInfo(FailureCode.CONTROLLER_FAILED if error_code in {None, 0} else FailureCode.UNKNOWN, backend_code=error_code)
