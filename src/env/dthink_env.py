import os
import argparse
from typing import Any, Dict, Optional
import numpy as np
import quaternion
import habitat
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from .habitat_extensions import measure

from .habitat_extensions import SensorGeometry
from ..utils import env_registry

@env_registry.register(aliases=("dthink_base",))
class DThinkEnv:
    """Habitat-based navigation environment wrapper for DThinkVLN."""
    ID2NAME = {
        HabitatSimActions.stop: "STOP",
        HabitatSimActions.move_forward: "MOVE_FORWARD",
        HabitatSimActions.turn_left: "TURN_LEFT",
        HabitatSimActions.turn_right: "TURN_RIGHT",
    }
    NAME2ID = {v: k for k, v in ID2NAME.items()}

    def __init__(self, cfg_path, split=None, gpu_device_id: Optional[int] = None):
        """
        Build the Habitat environment from a config file and optionally override
        the dataset split before instantiation.
        """
        assert os.path.exists(cfg_path), f"Configuration not found: {cfg_path}"
        self.cfg = get_config(cfg_path)

        # Override runtime config if specified
        if split is not None or gpu_device_id is not None:
            with read_write(self.cfg):
                # habitat-lab stores split under habitat.dataset.split
                if hasattr(self.cfg.habitat, "dataset") and hasattr(self.cfg.habitat.dataset, "split"):
                    if split is not None:
                        self.cfg.habitat.dataset.split = split
                elif split is not None:
                    raise AttributeError("Field habitat.dataset.split not found in config")

                # Select simulator gpu per-rank in distributed rollout.
                if (
                    hasattr(self.cfg.habitat, "simulator")
                    and hasattr(self.cfg.habitat.simulator, "habitat_sim_v0")
                    and hasattr(self.cfg.habitat.simulator.habitat_sim_v0, "gpu_device_id")
                ):
                    if gpu_device_id is not None:
                        self.cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = int(gpu_device_id)
                elif gpu_device_id is not None:
                    raise AttributeError("Field habitat.simulator.habitat_sim_v0.gpu_device_id not found in config")

        self.env: habitat.Env = habitat.Env(self.cfg)
        self.sg: SensorGeometry = SensorGeometry(self)

    def get_episode_list(self):
        """Return all episode ids in the current dataset."""
        return [ep.episode_id for ep in self.env._dataset.episodes]

    def set_episode(self, episode_id):
        """Set the current episode by id, reset the env, and return the first observation."""
        episode_id = str(episode_id)
        all_eps = {str(ep.episode_id): ep for ep in self.env._dataset.episodes}
        if episode_id not in all_eps:
            raise ValueError(f"Episode ID {episode_id} does not exist")
        self.env.current_episode = all_eps[episode_id]
        obs = self.env.reset()  # Reset clears metrics
        self._last_obs = self._postprocess_obs(obs)
        self.update()
        return self._last_obs

    def slice_episodes(self, start_idx: int, end_idx: int):
        """Slice the dataset episodes to [start_idx, end_idx) and rebuild the iterator."""
        eps = self.env._dataset.episodes
        n = len(eps)
        if not (0 <= start_idx < end_idx <= n):
            raise ValueError(
                f"Invalid range [{start_idx}, {end_idx}) for {n} episodes")

        self.env._dataset.episodes = eps[start_idx:end_idx]

        it_opts = self.cfg.habitat.environment.iterator_options
        self.env._episode_iterator = self.env._dataset.get_episode_iterator(
            shuffle=bool(getattr(it_opts, "shuffle", False)),
            group_by_scene=bool(getattr(it_opts, "group_by_scene", False)),
            max_scene_repeat_steps=int(
                getattr(it_opts, "max_scene_repeat_steps", 0)),
            num_episode_sample=getattr(it_opts, "num_episode_sample", None),
        )

        self.env._episode_from_iter_on_reset = True

        if hasattr(self.env, "_current_episode_index"):
            self.env._current_episode_index = 0

        print(
            f"[Env] Episodes sliced to [{start_idx}:{end_idx}), count={len(self.env._dataset.episodes)}")

    def update(self) -> None:
        """Refresh cached episode metadata and metrics."""
        self._episode_dict = self.env.current_episode.__dict__.copy()
        self._metrics = self.env.get_metrics()
        self._eps_over = self.env.episode_over
        self._traj = None

    def reset(self, next=True) -> Dict[str, Any]:
        """
        Reset the environment and return the first observation.
        If next is True, advance to the next episode from the iterator.
        """
        self.env._episode_from_iter_on_reset = next
        obs = self.env.reset()
        self._last_obs = self._postprocess_obs(obs)
        self.update()
        return self._last_obs

    def step(self, action: Any) -> Dict[str, Any]:
        """Take a step with the provided action and return the latest observation."""
        act_cmd = self._format_action(action)
        obs = self.env.step(act_cmd)
        self._last_obs = self._postprocess_obs(obs)
        self.update()
        return self._last_obs
    
    def snap_agent_to_navmesh(self, agent_id: int = 0):
        """
        Snap the agent to the nearest navigable location if it is off the navmesh.
        Returns (did_snap, new_pos).
        """
        sim = self.env.sim
        pf = sim.pathfinder

        # Read current agent state
        st = sim.get_agent_state(agent_id)
        cur_pos = np.asarray(st.position, dtype=np.float32)

        # Bail out if already navigable
        if pf.is_navigable(cur_pos):
            return False, cur_pos

        # Snap to nearest navigable point
        snapped = pf.snap_point(cur_pos)
        if snapped is None:
            raise ValueError(
                f"Agent position {cur_pos.tolist()} cannot be snapped to a navigable area (navmesh may be disconnected)"
            )

        snapped = np.asarray(snapped, dtype=np.float32)

        # Preserve rotation and update only the position
        sim.set_agent_state(snapped, st.rotation, agent_id=agent_id)
        return True, snapped
    
    def point_vln_step(self, action, sg = None, max_steps=500, goal_radius=0.2):
        """
        Execute a structured high-level command (TURN, MOVE_TO, or CMD).
        MOVE_TO leverages SensorGeometry to project a pixel to world coordinates.
        """
        if action is None:
            return
        
        if action["type"] == "TURN":
            for _ in range(action["times"]):
                self.step(action["action"])
        elif action["type"] == "MOVE_TO":
            uv = action["action"]
            if sg is None:
                sg = self.sg
            sensor_name = action.get("sensor", "depth")
            try:
                wp = sg.pixel_to_world(min(round(uv[0]), 279), min(round(uv[1]), 279), sensor_name=sensor_name, convention='z')
            except:
                wp = sg.pixel_to_world(min(round(uv[0]), 279), min(round(uv[1]), 279), sensor_name=sensor_name.replace("rgb", "depth"), convention='z')
            self.move_to_waypoint(wp, max_steps=max_steps, goal_radius=goal_radius)
        elif action["type"] == "CMD":
            self.step(action["action"])

    def _ensure_navigable(self, pos: np.ndarray) -> np.ndarray:
        """Return a navmesh-valid position, snapping to the nearest navigable point or raising if impossible."""
        sim = self.env.sim
        pf = sim.pathfinder
        pos = np.asarray(pos, dtype=np.float32)
        if pf.is_navigable(pos):
            return pos
        snapped = pf.snap_point(pos)
        if snapped is None:
            raise ValueError(f"Target point {pos.tolist()} cannot be snapped to a navigable area (navmesh disconnected or invalid)")
        return np.asarray(snapped, dtype=np.float32)

    def predict_next_action_to(
        self,
        waypoint: tuple,
        goal_radius: float = 0.2,
        return_name: bool = True,
        snap_to_navmesh: bool = True,
        allow_stop: bool = True,
    ):
        """
        Predict the next low-level action that moves the agent toward the waypoint
        using a shortest-path follower. Optionally snaps the goal to navmesh and
        returns either the action name or id.
        """
        sim = self.env.sim

        goal = np.asarray(waypoint, dtype=np.float32)
        if snap_to_navmesh:
            goal = self._ensure_navigable(goal)

        agent_pos = self.position

        geodesic = sim.geodesic_distance(agent_pos, goal)
        if not np.isfinite(geodesic):
            return None

        if np.linalg.norm(agent_pos - goal) <= goal_radius:
            if allow_stop:
                act = HabitatSimActions.stop
                return self.ID2NAME.get(act, "STOP") if return_name else int(act)
            return None

        follower = ShortestPathFollower(
            sim, goal_radius=goal_radius, return_one_hot=False)
        act = follower.get_next_action(goal)

        if act is None:
            if np.linalg.norm(self.position - goal) <= goal_radius and allow_stop:
                act = HabitatSimActions.stop
            else:
                return None

        return self.ID2NAME.get(act, str(act)) if return_name else int(act)

    def move_to_waypoint(
        self,
        waypoint: tuple,
        goal_radius: float = 0.2,
        max_steps: int = 500,
        snap_to_navmesh: bool = True
    ):
        """
        Move toward a waypoint by repeatedly calling predict_next_action_to until
        reaching the goal or hitting max_steps. Returns (path_points, actions).
        """
        path_points = [self.position]
        path_actions = ["START"]

        for _ in range(max_steps):
            action = self.predict_next_action_to(
                waypoint,
                goal_radius=goal_radius,
                return_name=True,
                snap_to_navmesh=snap_to_navmesh,
                allow_stop=True
            )

            if action is None or action.upper() == "STOP":
                break
            self.step(action)

            path_points.append(self.position)
            path_actions.append(action)

        return np.array(path_points), path_actions

    def move_to_end(
        self,
        reset: bool = True,
        goal_radius: float = 0.2,
        max_steps: int = 500,
        snap_to_navmesh: bool = True
    ):
        """
        Follow the reference_path of the current episode to the end, optionally
        resetting first. Returns concatenated waypoints and action names.
        """
        if reset:
            self.reset(False)
        reference_path = self.episode_dict["reference_path"]
        wps_list = [self.position.reshape(1, -1)]
        act_list = ["START"]
        for wp in reference_path[1:]:
            w, a = self.move_to_waypoint(
                wp, goal_radius, max_steps, snap_to_navmesh)
            wps_list.append(w[1:])
            act_list.extend(a[1:])
        wps_list.append(wps_list[-1][-1:])
        act_list.append("STOP")
        wps_list = np.concatenate(wps_list)

        return wps_list, act_list

    def path_between(
        self,
        start: Any,
        end: Any,
        snap_to_navmesh: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute shortest path between two coordinates using Habitat pathfinder.

        Returns a dict with:
        - found: bool
        - distance: geodesic distance (inf if not found)
        - points: path points as float32 array with shape (K, 3)
        - start: possibly snapped start point
        - end: possibly snapped end point
        """
        start_pt = np.asarray(start, dtype=np.float32).reshape(-1)[:3]
        end_pt = np.asarray(end, dtype=np.float32).reshape(-1)[:3]

        if snap_to_navmesh:
            start_pt = self._ensure_navigable(start_pt)
            end_pt = self._ensure_navigable(end_pt)

        try:
            import habitat_sim

            shortest_path = habitat_sim.ShortestPath()
        except Exception as exc:
            raise ImportError("`habitat_sim.ShortestPath` is required to compute shortest paths.") from exc

        shortest_path.requested_start = start_pt
        shortest_path.requested_end = end_pt
        found = self.env.sim.pathfinder.find_path(shortest_path)
        distance = float(getattr(shortest_path, "geodesic_distance", np.inf))
        points = np.asarray(getattr(shortest_path, "points", []), dtype=np.float32)
        if points.size == 0:
            points = np.empty((0, 3), dtype=np.float32)
        else:
            points = points.reshape(-1, 3).astype(np.float32, copy=False)

        if (not found) or (not np.isfinite(distance)):
            return {
                "found": False,
                "distance": float("inf"),
                "points": np.empty((0, 3), dtype=np.float32),
                "start": start_pt,
                "end": end_pt,
            }
        return {
            "found": True,
            "distance": distance,
            "points": points,
            "start": start_pt,
            "end": end_pt,
        }

    def obstacle_distance(
        self,
        point: Optional[Any] = None,
        snap_to_navmesh: bool = False,
        max_search_radius: float = 1.0,
    ) -> float:
        """
        Return distance from a world coordinate to the closest obstacle surface.

        Uses Habitat pathfinder's obstacle-distance query API.
        """
        if point is None:
            query_point = np.asarray(self.position, dtype=np.float32).reshape(-1)[:3]
        else:
            query_point = np.asarray(point, dtype=np.float32).reshape(-1)[:3]
        if snap_to_navmesh:
            query_point = self._ensure_navigable(query_point)

        if max_search_radius <= 0:
            raise ValueError(f"`max_search_radius` must be positive, got {max_search_radius}.")

        pf = self.env.sim.pathfinder
        try:
            dist = pf.distance_to_closest_obstacle(query_point, float(max_search_radius))
        except TypeError:
            # Backward compatibility for versions exposing a single-arg signature.
            dist = pf.distance_to_closest_obstacle(query_point)
        return float(dist)

    def traj_from_ref(
        self,
        merge_eps: float = 1e-2,
        interp_interval: Optional[float] = None,
        snap_to_navmesh: bool = True,
        strict: bool = True,
    ) -> np.ndarray:
        """
        Build a dense expert trajectory by planning shortest paths between
        consecutive keypoints in the current episode `reference_path`.

        This function does not reset/step the environment and does not move agent.
        Consecutive points whose L2 distance is smaller than `merge_eps` are merged.
        If `interp_interval` is not None, linearly interpolate so each consecutive
        spacing is at most `interp_interval` meters.
        """
        reference_path = self.episode_dict.get("reference_path")

        if not reference_path:
            return np.empty((0, 3), dtype=np.float32)
        expert_points = []
        if interp_interval is not None and interp_interval <= 0:
            raise ValueError(f"`interp_interval` must be positive when enabled, got {interp_interval}.")

        def _append_point(point: Any):
            p = np.asarray(point, dtype=np.float32).reshape(-1)[:3]
            if len(expert_points) == 0:
                expert_points.append(p)
                return

            prev = expert_points[-1]
            dist = float(np.linalg.norm(p - prev))
            if dist < merge_eps:
                return

            if interp_interval is not None and dist > interp_interval:
                # Split [prev, p] into n equal segments so each segment length <= interp_interval.
                n_segments = int(np.ceil(dist / interp_interval))
                for i in range(1, n_segments):
                    interp_p = prev + (p - prev) * (i / n_segments)
                    if np.linalg.norm(interp_p - expert_points[-1]) >= merge_eps:
                        expert_points.append(interp_p.astype(np.float32, copy=False))

            if np.linalg.norm(p - expert_points[-1]) >= merge_eps:
                expert_points.append(p)

        if len(reference_path) == 1:
            _append_point(reference_path[0])
            return np.stack(expert_points, axis=0).astype(np.float32, copy=False)

        for seg_idx in range(len(reference_path) - 1):
            seg_start = np.asarray(reference_path[seg_idx], dtype=np.float32).reshape(-1)[:3]
            seg_end = np.asarray(reference_path[seg_idx + 1], dtype=np.float32).reshape(-1)[:3]
            path_result = self.path_between(seg_start, seg_end, snap_to_navmesh=snap_to_navmesh)
            seg_points = path_result["points"]

            if not path_result["found"] or len(seg_points) == 0:
                msg = (
                    f"Failed to find shortest path for segment {seg_idx}: "
                    f"start={path_result['start'].tolist()}, end={path_result['end'].tolist()}"
                )
                if strict:
                    raise RuntimeError(msg)
                # Fallback: at least keep keypoint connectivity in output.
                seg_points = np.stack([path_result["start"], path_result["end"]], axis=0)

            if seg_idx > 0 and len(seg_points) > 0:
                seg_points = seg_points[1:]

            for p in seg_points:
                _append_point(p)

        if len(expert_points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.stack(expert_points, axis=0).astype(np.float32, copy=False)

    def expert_dist(
        self,
        current_position: Optional[np.ndarray] = None,
        snap_to_navmesh: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate current-to-goal expert distance using wrapped `self.traj`.

        For each point on expert trajectory, compute:
            connector_distance(current -> traj[i]) + remaining_traj_distance(i -> goal)
        and choose the minimum total distance.
        """
        traj = np.asarray(self.traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 3:
            raise ValueError(f"`self.traj` must be shape (N, 3+) with N>0, got {traj.shape}.")
        traj = traj[:, :3]

        if current_position is None:
            current = np.asarray(self.position, dtype=np.float32).reshape(-1)[:3]
        else:
            current = np.asarray(current_position, dtype=np.float32).reshape(-1)[:3]

        # Remaining polyline distance from each trajectory point to the goal point traj[-1].
        remaining = np.zeros(traj.shape[0], dtype=np.float32)
        if traj.shape[0] > 1:
            seg_len = np.linalg.norm(traj[1:] - traj[:-1], axis=1).astype(np.float32)
            remaining[:-1] = np.cumsum(seg_len[::-1], dtype=np.float32)[::-1]
        best_idx = -1
        best_total_dist = float("inf")
        best_connector_dist = float("inf")
        best_remaining_dist = float("inf")
        best_connector_points = np.empty((0, 3), dtype=np.float32)

        for idx, point in enumerate(traj):
            path_result = self.path_between(current, point, snap_to_navmesh=snap_to_navmesh)
            connector_dist = float(path_result["distance"])
            if not path_result["found"]:
                continue
            remaining_dist = float(remaining[idx])
            total_dist = connector_dist + remaining_dist
            if total_dist < best_total_dist:
                best_idx = idx
                best_total_dist = total_dist
                best_connector_dist = connector_dist
                best_remaining_dist = remaining_dist
                best_connector_points = path_result["points"]

        if best_idx < 0:
            msg = "Failed to find shortest path from current position to any expert trajectory point."
            if strict:
                raise RuntimeError(msg)
            return {
                "expert_distance": float("inf"),
                "connector_distance": float("inf"),
                "remaining_traj_distance": float("inf"),
                "nearest_traj_index": -1,
                "nearest_traj_point": None,
                "connector_path_points": np.empty((0, 3), dtype=np.float32),
                "goal_point": traj[-1].astype(np.float32, copy=False),
            }

        total_expert_dist = float(best_total_dist)

        return {
            "expert_distance": total_expert_dist,
            "connector_distance": float(best_connector_dist),
            "remaining_traj_distance": float(best_remaining_dist),
            "nearest_traj_index": int(best_idx),
            "nearest_traj_point": traj[best_idx].astype(np.float32, copy=False),
            "connector_path_points": best_connector_points,
            "goal_point": traj[-1].astype(np.float32, copy=False),
        }

    def list_available_sensors(self):
        """Return the list of available sensor names on the current agent."""
        st = self.env.sim.get_agent_state()
        return list(st.sensor_states.keys())

    def get_sensor_pose(self, sensor_name: Optional[str] = None):
        """
        Get pose of a specific sensor or all sensors.
        Returns position (np.array) and rotation (quaternion as np.array).
        """
        st = self.env.sim.get_agent_state()
        sensors = st.sensor_states

        def _pose(ss):
            q = ss.rotation
            R = quaternion.as_rotation_matrix(q)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arcsin(-R[2, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])

            return {
                "position": np.array(ss.position, dtype=np.float32),
                "rotation": np.array([q.w, q.x, q.y, q.z], dtype=np.float32),
                "euler_rotation": np.array([roll, pitch, yaw], dtype=np.float32),
            }

        if sensor_name is not None:
            if sensor_name not in sensors:
                raise ValueError(f"Sensor '{sensor_name}' not found; available: {list(sensors.keys())}")
            return _pose(sensors[sensor_name])

        return {name: _pose(ss) for name, ss in sensors.items()}
    
    @property
    def traj(self) -> np.ndarray:
        if self._traj is None: 
            self._traj = self.traj_from_ref(merge_eps=0.01, interp_interval=0.2, snap_to_navmesh=True, strict=True)
        return self._traj
    
    @property
    def position(self) -> np.ndarray:
        """Agent position in world coordinates (x, y, z)."""
        return np.array(self.env.sim.get_agent_state().position, dtype=np.float32)
    
    @property
    def rotation(self) -> np.ndarray:
        """Agent rotation in quaternion (w, x, y, z)."""
        q = self.env.sim.get_agent_state().rotation
        return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
    
    @property
    def rotation_euler(self) -> np.ndarray:
        """Agent rotation in euler (roll, pitch, yaw)."""
        R = quaternion.as_rotation_matrix(self.env.sim.get_agent_state().rotation)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        return np.array([roll, pitch, yaw], dtype=np.float32)

    @property
    def observations(self) -> Optional[Dict[str, Any]]:
        """Latest cached observation after the last reset/step."""
        return self._last_obs

    @property
    def episode_over(self) -> bool:
        """Whether the current episode has ended."""
        return self._eps_over

    @property
    def metrics(self) -> Optional[Dict[str, Any]]:
        """Latest cached metrics from the environment."""
        return self._metrics

    @property
    def action_space(self):
        """Available action names for this environment."""
        return list(self.NAME2ID.keys())

    @property
    def episode_dict(self) -> str:
        """Metadata of the current episode cached as a dict."""
        return self._episode_dict

    def close(self) -> None:
        """Close the underlying Habitat environment."""
        self.env.close()

    def _format_action(self, a: Any) -> Dict[str, Any]:
        """
        Normalize different action representations into the dict expected by
        habitat.Env.step. Supports dict, int, and string inputs.
        """
        if isinstance(a, dict):
            return a

        if isinstance(a, int):
            return {"action": a}

        if isinstance(a, str):
            name = a.strip().upper()
            if name in self.NAME2ID:
                return {"action": self.NAME2ID[name]}
            return {"action": a.strip().lower()}

        raise TypeError(f"Unsupported action type: {type(a)}")

    def _postprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for observation post-processing. Currently returns the input unchanged."""
        return obs
