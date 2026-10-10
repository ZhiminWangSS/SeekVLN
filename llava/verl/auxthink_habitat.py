"""Persistent, sharded Habitat backend used by AuxThink VERL AgentLoops."""

from __future__ import annotations

import gzip
import json
import math
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import habitat_sim
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class

from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.config.default import get_config


ENV_NAME = "AuxThinkVerlHabitatEnv"
EVAL_ROOT = Path(__file__).resolve().parents[2] / "evaluation"


def expert_primitive_limit(expert_primitive_steps: int, ratio: float) -> int:
    if int(expert_primitive_steps) < 1 or float(ratio) < 1.0:
        raise ValueError("expert primitive length must be positive and ratio at least 1.0")
    return max(1, int(math.floor(int(expert_primitive_steps) * float(ratio))))


def _register_environment() -> None:
    from vlnce_baselines.common.environments import VLNCECollectorEnv

    if baseline_registry.get_env(ENV_NAME) is not None:
        return

    @baseline_registry.register_env(name=ENV_NAME)
    class AuxThinkVerlHabitatEnv(VLNCECollectorEnv):
        def skip_episode_iterator(self, count):
            """Advance the same iterator as reset, without loading skipped scenes."""
            for _ in range(max(0, int(count))):
                next(self._env.episode_iterator)

        def get_seekvln_panorama(self):
            agent_state = self._env.sim.get_agent_state()
            position = agent_state.position
            base_rotation = agent_state.rotation
            views = {}
            for name, angle in (
                ("left", np.pi / 2.0),
                ("forward", 0.0),
                ("right", -np.pi / 2.0),
                ("back", np.pi),
            ):
                delta = habitat_sim.utils.quat_from_angle_axis(angle, habitat_sim.geo.UP)
                rotation = base_rotation * delta
                views[name] = self._env.sim.get_observations_at(
                    position=position,
                    rotation=rotation,
                    keep_agent_at_new_pose=False,
                )
            return views

        def get_geodesic_distance_to_goal(self):
            """Return the current goal distance computed by Habitat Pathfinder."""
            position = self._env.sim.get_agent_state().position
            episode = self._env.current_episode
            goals = [goal.position for goal in episode.goals]
            if not goals:
                return float("inf")
            return float(self._env.sim.geodesic_distance(position, goals, episode))

        def get_topdown_visualization(self):
            from habitat.utils.visualizations import maps as hm
            from evaluation.habitat_extensions.maps import get_top_down_map, colorize_topdown_map
            sim = self._env.sim
            ep = self._env.current_episode
            m = colorize_topdown_map(get_top_down_map(sim, 512, 0.05))
            st = sim.get_agent_state()
            pts = [st.position]
            start = np.asarray(ep.start_position)
            goal = np.asarray(ep.goals[0].position)
            def grid(p):
                y, x = hm.to_grid(float(p[2]), float(p[0]), m.shape[:2], sim)
                return int(x), int(y)
            return {"map": m, "agent": grid(st.position), "start": grid(start), "goal": grid(goal)}


class AuxThinkVerlHabitatBackend:
    """One persistent Habitat iterator assigned to one AgentLoop worker shard."""

    def __init__(
        self,
        exp_config: str,
        split: str,
        dataset_path: str,
        gt_path: str,
        scenes_dir: str,
        primitive_step_limit_ratio: Optional[float] = 1.5,
        num_chunks: int = 1,
        chunk_idx: int = 0,
        skip_episodes: int = 0,
        episode_ids: Optional[list[str]] = None,
    ):
        _register_environment()
        if not 0 <= int(chunk_idx) < int(num_chunks):
            raise ValueError(f"invalid Habitat shard {chunk_idx}/{num_chunks}")
        previous_cwd = Path.cwd()
        try:
            os.chdir(EVAL_ROOT)
            config_path = Path(exp_config)
            if not config_path.is_absolute():
                config_path = (EVAL_ROOT.parent / config_path).resolve()
            opts = [
                    "TASK_CONFIG.DATASET.SPLIT",
                    str(split),
                    "TASK_CONFIG.DATASET.DATA_PATH",
                    str(dataset_path),
                    "TASK_CONFIG.DATASET.SCENES_DIR",
                    str(scenes_dir),
                    "TASK_CONFIG.TASK.NDTW.GT_PATH",
                    str(gt_path),
                    "TASK_CONFIG.TASK.NDTW.SPLIT",
                    str(split),
                ]
            if episode_ids:
                opts.extend(
                    ["TASK_CONFIG.DATASET.EPISODES_ALLOWED", [str(value) for value in episode_ids]]
                )
            config = get_config(str(config_path), opts=opts)
            resolved_split = str(config.TASK_CONFIG.DATASET.SPLIT)
            with gzip.open(str(gt_path).format(split=resolved_split), "rt", encoding="utf-8") as handle:
                ground_truth = json.load(handle)
            self.expert_primitive_lengths = {
                str(episode_id): len(trajectory["actions"])
                for episode_id, trajectory in ground_truth.items()
            }
            self.primitive_step_limit_ratio = (
                None if primitive_step_limit_ratio is None else float(primitive_step_limit_ratio)
            )
            config.defrost()
            config.ENV_NAME = ENV_NAME
            config.NUM_ENVIRONMENTS = 1
            config.TASK_CONFIG.DATASET.NUM_CHUNKS = int(num_chunks)
            config.TASK_CONFIG.DATASET.CHUNK_IDX = int(chunk_idx)
            config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
            config.freeze()
            self.envs = construct_envs_auto_reset_false(config, get_env_class(ENV_NAME))
            if int(skip_episodes) > 0:
                self.envs.call_at(0, "skip_episode_iterator", {"count": int(skip_episodes)})
        finally:
            os.chdir(previous_cwd)
        self.rgb_history = []
        # ``asyncio.to_thread`` cannot stop a running Habitat call when its
        # coroutine is cancelled by the rollout timeout.  Serialize all
        # public backend operations at the thread level so a replacement
        # rollout waits for that call instead of concurrently using the same
        # VectorEnv pipe.
        self._thread_lock = threading.RLock()
        self.current_observation = None
        self.initial_distance = None
        self.initial_geodesic_distance = None
        self.primitive_steps = 0
        self.expert_primitive_steps = 0
        self.primitive_step_limit = None
        self.truncated = False
        self.truncation_reason = None

    @property
    def episode_id(self) -> str:
        with self._thread_lock:
            return str(self.envs.current_episodes()[0].episode_id)

    def _decorate(self, observation):
        result = dict(observation)
        current = np.asarray(result["rgb"])
        result["auxthink_instruction"] = self.envs.current_episodes()[0].instruction.instruction_text
        result["auxthink_rgb_history"] = [*self.rgb_history, current]
        return result

    def reset(self):
        with self._thread_lock:
            self.rgb_history = []
            self.current_observation = self.envs.reset()[0]
            initial_metrics = self._raw_metrics()
            self.initial_distance = initial_metrics.get("distance_to_goal")
            self.initial_geodesic_distance = initial_metrics.get("geodesic_distance_to_goal")
            episode_id = self.episode_id
            if episode_id not in self.expert_primitive_lengths:
                raise KeyError(f"GT actions missing for episode {episode_id}")
            self.primitive_steps = 0
            self.expert_primitive_steps = int(self.expert_primitive_lengths[episode_id])
            if self.primitive_step_limit_ratio is None:
                self.primitive_step_limit = None
            else:
                self.primitive_step_limit = expert_primitive_limit(
                    self.expert_primitive_steps, self.primitive_step_limit_ratio
                )
            self.truncated = False
            self.truncation_reason = None
            return self._decorate(self.current_observation)

    def step(self, action):
        with self._thread_lock:
            final = None
            primitives = list(action) if isinstance(action, (list, tuple)) else [int(action)]
            for primitive in primitives:
                if self.primitive_step_limit is not None and self.primitive_steps >= self.primitive_step_limit:
                    self.truncated = True
                    self.truncation_reason = "primitive_limit"
                    break
                self.rgb_history.append(np.asarray(self.current_observation["rgb"]))
                final = self.envs.step([int(primitive)])[0]
                self.current_observation = final[0]
                self.primitive_steps += 1
                if final[2]:
                    break
            if final is None:
                observation, habitat_done, info = self.current_observation, False, self._raw_metrics()
            else:
                observation, _, habitat_done, info = final
            if (
                not habitat_done
                and self.primitive_step_limit is not None
                and self.primitive_steps >= self.primitive_step_limit
            ):
                self.truncated = True
                self.truncation_reason = "primitive_limit"
            done = bool(habitat_done or self.truncated)
            info = dict(info or {})
            info.update(self._limit_metrics())
            return self._decorate(observation), done, info

    def _raw_metrics(self):
        metrics = dict(self.envs.call_at(0, "get_info", {"observations": self.current_observation or {}}))
        metrics["geodesic_distance_to_goal"] = float(
            self.envs.call_at(0, "get_geodesic_distance_to_goal", {})
        )
        return metrics

    def get_metrics(self):
        with self._thread_lock:
            metrics = self._raw_metrics()
            metrics["initial_distance"] = self.initial_distance
            metrics["initial_geodesic_distance"] = self.initial_geodesic_distance
            metrics.update(self._limit_metrics())
            return metrics

    def get_topdown_visualization(self):
        with self._thread_lock:
            info = self.envs.call_at(0, "get_info", {"observations": self.current_observation or {}})
            td = info["top_down_map_vlnce"]
            from evaluation.habitat_extensions.maps import colorize_topdown_map
            return {"map": colorize_topdown_map(td["map"], td.get("fog_of_war_mask")), "agent": td["agent_map_coord"]}

    def get_video_observation(self):
        with self._thread_lock:
            views = self.envs.call_at(0, "get_seekvln_panorama", {})
            obs = dict(views.get("forward", {}))
            for name, value in views.items():
                if "rgb" in value:
                    obs["rgb" if name == "forward" else f"rgb_{name}"] = value["rgb"]
            obs["instruction"] = {"text": self.envs.current_episodes()[0].instruction.instruction_text}
            return obs

    def get_seekvln_panorama(self):
        with self._thread_lock:
            observations = self.envs.call_at(0, "get_seekvln_panorama", {})
            return {
                name: np.asarray(observation["rgb"])
                for name, observation in dict(observations).items()
            }

    def _limit_metrics(self):
        return {
            "primitive_steps": int(self.primitive_steps),
            "expert_primitive_steps": int(self.expert_primitive_steps),
            "primitive_step_limit": None
            if self.primitive_step_limit is None
            else int(self.primitive_step_limit),
            "auxthink_truncated": bool(self.truncated),
            "auxthink_truncation_reason": self.truncation_reason,
        }

    def close(self):
        with self._thread_lock:
            self.envs.close()
