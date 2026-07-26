from typing import Any, Dict, Optional

from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.tasks.nav.nav import DistanceToGoal

@registry.register_measure
class OracleNavigationError(Measure):
    """
    Oracle Navigation Error (ONE):
    This measure calculates the minimum geodesic distance between the agent's 
    path and the goal location. It represents the closest the agent has been 
    to the goal during its trajectory.

    Attributes:
        cls_uuid (str): Unique identifier for this measure.
    """

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    """
    Oracle Success Rate (OSR):
    This measure calculates whether the agent's closest distance to the goal 
    (Oracle Navigation Error) is within a specified threshold (goal radius). 
    OSR = 1 if ONE <= goal_radius, otherwise 0.

    Attributes:
        cls_uuid (str): Unique identifier for this measure.
        _config (Any): Configuration for the measure.
    """

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any):
        # print(f"in oracle success init: args = {args}, kwargs = {kwargs}")
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = float(self._metric or d < 3.0)
