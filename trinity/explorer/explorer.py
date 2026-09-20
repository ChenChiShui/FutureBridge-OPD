# -*- coding: utf-8 -*-
"""The explorer module"""
from __future__ import annotations

import asyncio
import math
import os
import time
import traceback
from collections import deque
from typing import List, Optional

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from trinity.buffer.buffer import get_buffer_reader
from trinity.buffer.pipelines.experience_pipeline import ExperiencePipeline
from trinity.buffer.task_scheduler import get_taskset_scheduler
from trinity.common.config import Config
from trinity.common.constants import (
    ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
    RunningStatus,
    SyncMethod,
    SyncStyle,
)
from trinity.common.models import create_inference_models
from trinity.explorer.scheduler import Scheduler
from trinity.manager.state_manager import StateManager
from trinity.manager.synchronizer import Synchronizer
from trinity.utils.annotations import Experimental
from trinity.utils.log import get_logger
from trinity.utils.monitor import MONITOR, gather_eval_metrics, gather_metrics
from trinity.utils.plugin_loader import load_plugins
from trinity.utils.timer import Timer


class Explorer:
    """Responsible for exploring the taskset."""

    def __init__(self, config: Config):
        self.logger = get_logger(config.explorer.name, in_ray_actor=True)
        load_plugins()
        self.state = StateManager(
            path=config.checkpoint_job_dir, explorer_name=config.explorer.name, config=config
        )
        explorer_state = self.state.load_explorer()
        self.explore_step_num = explorer_state.get("latest_iteration", 0)
        self.last_sync_step = self.explore_step_num
        self.last_monitored_step = self.explore_step_num
        self.synchronizer = Synchronizer.get_actor(config)
        self.config = config
        self.model_type = config.explorer.rollout_model.engine_type
        self.models, self.auxiliary_models = create_inference_models(config)
        self.experience_pipeline = self._init_experience_pipeline()
        self.taskset = (
            get_taskset_scheduler(explorer_state=explorer_state, config=config)
            if self.config.mode not in {"bench", "serve"}
            else None
        )
        self.scheduler = None
        self.monitor = MONITOR.get(self.config.monitor.monitor_type)(
            project=self.config.project,
            group=self.config.group,
            name=self.config.name,
            role=self.config.explorer.name,
            config=config,
        )
        self.detailed_stats = config.monitor.detailed_stats
        if config.explorer.over_rollout.ratio > 0.0:
            self.min_wait_num = math.ceil(
                config.buffer.batch_size * (1 - config.explorer.over_rollout.ratio)
            )
            self.logger.info(
                f"Over rollout is enabled. Explorer will only wait for {self.min_wait_num} tasks in each step."
            )
        else:
            self.min_wait_num = None
        self.use_nccl_sync = self.config.synchronizer.sync_method == SyncMethod.NCCL
        self.pending_eval_tasks = deque()

        # For checkpoint weights update
        # Use explorer to periodically load the latest model weights and
        # boradcast to all rollout models
        self.enable_lora = self.config.explorer.rollout_model.enable_lora
        self.model_version = -1
        self.last_sync_successful = True
        self.teacher_regularization, self.teacher_update_rate = (
            self._get_teacher_regularization_config()
        )
        self.enable_teacher_ema = self.teacher_regularization == "ema"
        self.teacher_ema_state_dict = None
        self.teacher_ema_metrics = {}
        self.eval_start_time = None
        self.explore_start_time = None
        self.logger.info("Finished initializing Explorer.")

    def _get_teacher_regularization_config(self) -> tuple[str, float]:
        taskset = self.config.buffer.explorer_input.taskset
        workflow_args = taskset.workflow_args if taskset is not None else {}
        teacher_regularization = str(workflow_args.get("teacher_regularization", "")).lower()
        teacher_update_rate = float(workflow_args.get("teacher_update_rate", 0.01))
        return teacher_regularization, teacher_update_rate

    def _clone_state_dict(self, state_dict: dict) -> dict:
        cloned = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().clone()
            else:
                cloned[key] = value
        return cloned

    def _update_teacher_ema_state(self, student_state_dict: dict) -> None:
        if self.teacher_ema_state_dict is None:
            self.teacher_ema_state_dict = self._clone_state_dict(student_state_dict)
            return
        ema_rate = self.teacher_update_rate
        for key, student_weight in student_state_dict.items():
            teacher_weight = self.teacher_ema_state_dict.get(key)
            if teacher_weight is None or not torch.is_tensor(student_weight):
                self.teacher_ema_state_dict[key] = (
                    student_weight.detach().clone()
                    if torch.is_tensor(student_weight)
                    else student_weight
                )
                continue
            if (not torch.is_tensor(teacher_weight)) or teacher_weight.shape != student_weight.shape:
                self.teacher_ema_state_dict[key] = student_weight.detach().clone()
                continue
            teacher_weight = teacher_weight.to(
                device=student_weight.device, dtype=student_weight.dtype
            )
            teacher_weight.mul_(1.0 - ema_rate).add_(student_weight, alpha=ema_rate)
            self.teacher_ema_state_dict[key] = teacher_weight

    def _compute_teacher_student_param_gap_metrics(self, student_state_dict: dict) -> dict:
        if self.teacher_ema_state_dict is None:
            return {}
        diff_sq_sum = 0.0
        student_sq_sum = 0.0
        abs_diff_sum = 0.0
        n_elements = 0

        for key, student_weight in student_state_dict.items():
            teacher_weight = self.teacher_ema_state_dict.get(key)
            if (
                teacher_weight is None
                or not torch.is_tensor(student_weight)
                or not torch.is_tensor(teacher_weight)
            ):
                continue
            if teacher_weight.shape != student_weight.shape:
                continue
            student_fp32 = student_weight.detach().float()
            teacher_fp32 = teacher_weight.detach().to(
                device=student_fp32.device, dtype=torch.float32
            )
            diff = teacher_fp32 - student_fp32
            diff_sq_sum += diff.pow(2).sum().item()
            student_sq_sum += student_fp32.pow(2).sum().item()
            abs_diff_sum += diff.abs().sum().item()
            n_elements += diff.numel()

        if n_elements == 0:
            return {}

        gap_l2 = diff_sq_sum**0.5
        student_l2 = student_sq_sum**0.5
        relative_l2 = gap_l2 / max(student_l2, 1e-12)
        mean_abs_gap = abs_diff_sum / n_elements
        return {
            "sdpo_teacher_ema/param_gap_l2": float(gap_l2),
            "sdpo_teacher_ema/param_gap_l2_relative": float(relative_l2),
            "sdpo_teacher_ema/param_gap_abs_mean": float(mean_abs_gap),
        }

    async def _sync_auxiliary_models(self, model_version: int) -> None:
        if not self.auxiliary_models:
            return
        await asyncio.gather(
            *[
                model.sync_model.remote(model_version, "student")
                for models in self.auxiliary_models
                for model in models
            ]
        )

    async def _sync_auxiliary_models_with_ema(self, model_version: int) -> None:
        if not self.auxiliary_models:
            return
        student_state_dict, _ = await self.synchronizer.get_model_state_dict.remote("student")
        if not isinstance(student_state_dict, dict):
            self.logger.warning(
                "Cannot build teacher EMA state dict from non-dict student state. "
                "Falling back to hard teacher sync."
            )
            await self._sync_auxiliary_models(model_version)
            return
        self._update_teacher_ema_state(student_state_dict)
        self.teacher_ema_metrics = self._compute_teacher_student_param_gap_metrics(
            student_state_dict
        )
        await self.synchronizer.set_named_model_state_dict.remote(
            "teacher_ema", self.teacher_ema_state_dict, model_version
        )
        await asyncio.gather(
            *[
                model.sync_model.remote(model_version, "teacher_ema")
                for models in self.auxiliary_models
                for model in models
            ]
        )

    async def setup_weight_sync_group(
        self, master_address: str, master_port: int, state_dict_meta: List = None
    ):
        base_offset = 1 if self.use_nccl_sync else 0
        world_size = (
            len(self.models) * self.config.explorer.rollout_model.tensor_parallel_size + base_offset
        )
        self.logger.info(
            f"Initialize process group for weight synchronization, "
            f"master_address={master_address}, master_port={master_port}, "
            f"world_size={world_size}, rank_offset={base_offset}"
        )
        # TODO: save state_dict in models
        refs = [
            model.init_process_group.remote(
                master_address=master_address,
                master_port=master_port,
                rank_offset=i * self.config.explorer.rollout_model.tensor_parallel_size
                + base_offset,
                world_size=world_size,
                group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                explorer_name=self.config.explorer.name,
                timeout=self.config.synchronizer.sync_timeout,
                state_dict_meta=state_dict_meta,
            )
            for i, model in enumerate(self.models)
        ]
        # Auxiliary models (e.g. teacher) need per-model process groups for weight sync
        for auxiliary_config, models in zip(
            self.config.explorer.auxiliary_models or [],
            self.auxiliary_models or [],
        ):
            if not models:
                continue
            aux_master_address, aux_master_port = await models[0].get_available_address.remote()
            aux_world_size = auxiliary_config.tensor_parallel_size
            for i, model in enumerate(models):
                refs.append(
                    model.init_process_group.remote(
                        master_address=aux_master_address,
                        master_port=aux_master_port,
                        rank_offset=i,
                        world_size=aux_world_size,
                        group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                        explorer_name=self.config.explorer.name,
                        timeout=self.config.synchronizer.sync_timeout,
                    )
                )
        await asyncio.gather(*refs)

    async def setup_model_level_weight_sync_group(self):
        """Setup process group for each model, only used in serve mode."""
        refs = []
        world_size = self.config.explorer.rollout_model.tensor_parallel_size
        for model in self.models:
            master_address, master_port = await model.get_available_address.remote()
            self.logger.info(
                f"Initialize process group for model weight synchronization, "
                f"master_address={master_address}, master_port={master_port}, "
                f"world_size={world_size}"
            )
            refs.append(
                model.init_process_group.remote(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=0,
                    world_size=world_size,
                    group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                    explorer_name=self.config.explorer.name,
                    timeout=self.config.synchronizer.sync_timeout,
                )
            )
        # Auxiliary models (e.g. teacher) need per-model process groups
        for auxiliary_config, models in zip(
            self.config.explorer.auxiliary_models, self.auxiliary_models
        ):
            if not models:
                continue
            aux_master_address, aux_master_port = await models[0].get_available_address.remote()
            aux_world_size = auxiliary_config.tensor_parallel_size
            self.logger.info(
                f"Initialize process group for auxiliary model weight synchronization, "
                f"master_address={aux_master_address}, master_port={aux_master_port}, "
                f"world_size={aux_world_size}"
            )
            for i, model in enumerate(models):
                refs.append(
                    model.init_process_group.remote(
                        master_address=aux_master_address,
                        master_port=aux_master_port,
                        rank_offset=i,
                        world_size=aux_world_size,
                        group_name=ROLLOUT_WEIGHT_SYNC_GROUP_NAME,
                        explorer_name=self.config.explorer.name,
                        timeout=self.config.synchronizer.sync_timeout,
                    )
                )
        await asyncio.gather(*refs)

    async def _checkpoint_weights_update(self, step_num: Optional[int] = None) -> int:
        self.logger.info(f"Start to update model weights from checkpoint at step {step_num}.")
        step_num = await self.synchronizer.set_model_state_dict_with_step_num.remote(step_num)
        await asyncio.gather(
            *[model.sync_model.remote(step_num, "student") for model in self.models]
        )
        # Only sync auxiliary models if teacher regularization is enabled
        if self.teacher_regularization:
            await self._sync_auxiliary_models(step_num)
        self.logger.info(f"Model weights updated to checkpoint at step {step_num}.")
        return step_num  # type: ignore

    async def _pull_latest_weights(self):
        self.logger.info("Start to pull latest model weights.")
        new_version = await self.synchronizer.wait_new_model_state_dict.remote(
            current_version=self.model_version,
            no_wait=(self.config.synchronizer.sync_style != SyncStyle.FIXED),
        )
        if new_version > self.model_version:
            if self.model_version != -1:
                self.logger.info(f"New model weights version: {new_version}")
                await asyncio.gather(
                    *[model.sync_model.remote(new_version, "student") for model in self.models]
                )
            # Only sync auxiliary models if teacher regularization is enabled
            # For OPD with fixed teacher, teacher_regularization should not be set
            if self.enable_teacher_ema and not self.use_nccl_sync:
                await self._sync_auxiliary_models_with_ema(new_version)
            elif self.enable_teacher_ema and self.use_nccl_sync:
                self.logger.warning(
                    "teacher_regularization=ema is not supported with NCCL sync. "
                    "Falling back to hard teacher sync."
                )
                await self._sync_auxiliary_models(new_version)
            elif self.teacher_regularization:
                # If teacher_regularization is set but not "ema", sync auxiliary models
                await self._sync_auxiliary_models(new_version)
            self.model_version = new_version
            self.last_sync_step = self.explore_step_num
            self.last_sync_successful = True
        else:
            self.logger.warning(
                f"No new model weights found, current version: {self.model_version}"
            )
            self.last_sync_successful = False

    async def _nccl_weights_update(self):
        new_version = await self.synchronizer.ready_to_nccl_sync.remote(
            "explorer", self.model_version
        )
        if new_version is None:
            self.logger.info("Trainer is not ready to sync weight. Skipping sync weight.")
            self.last_sync_successful = False
            return
        self.model_version = new_version
        await asyncio.gather(
            *[model.sync_model.remote(self.model_version, "student") for model in self.models]
        )
        # Only sync auxiliary models if teacher regularization is enabled
        if self.teacher_regularization:
            await self._sync_auxiliary_models(self.model_version)
        self.last_sync_step = self.explore_step_num
        self.last_sync_successful = True

    async def prepare(self) -> None:
        """Preparation before running."""
        try:
            # prepare experience pipeline
            if self.experience_pipeline:
                await self.experience_pipeline.prepare.remote()
            self.logger.info("Experience pipeline is ready.")
            # make sure all rollout models are ready
            run_api_ref = [model.prepare.remote() for model in self.models]
            run_api_ref.extend(
                model.prepare.remote() for models in self.auxiliary_models for model in models
            )
            await asyncio.gather(*run_api_ref)
            self.logger.info("All models are ready.")

            if not self.use_nccl_sync and self.model_type != "tinker":
                if self.config.mode == "serve":
                    # In serving mode, each engine will setup its own process group
                    await self.setup_model_level_weight_sync_group()
                else:
                    master_address, master_port = await self.models[
                        0
                    ].get_available_address.remote()
                    await self.setup_weight_sync_group(master_address, master_port)
            if self.config.mode != "serve":
                self.scheduler = Scheduler(self.config, self.models, self.auxiliary_models)
                await self.scheduler.start()
            if self.config.explorer.eval_on_startup and self.explore_step_num == 0:
                await self.eval()

            await self.synchronizer.set_explorer_status.remote(RunningStatus.REQUIRE_SYNC)
        except Exception as e:
            self.logger.error(f"Error during explorer preparation: {traceback.format_exc()}")
            await self.shutdown()
            raise e

    async def get_weight(self, name: str) -> torch.Tensor:
        """Get the weight of the loaded model (For checkpoint weights update)."""
        return self.state_dict[name]

    async def explore(self) -> str:
        """
        The timeline of the exploration process:
                 | <--------------------------------- one period -------------------------------------> |
        explorer | <---------------- step_1 --------------> |                                           |
                 |   | <---------------- step_2 --------------> |                                       |
                 |      ...                                                                             |
                 |          | <---------------- step_n ---------------> |                               |
                 |                  | <---------------------- eval --------------------> | <-- sync --> |
                 |--------------------------------------------------------------------------------------|
        trainer  | <-- idle --> | <-- step_1 --> | <-- step_2 --> | ... | <-- step_n --> | <-- sync --> |
        """
        while True:
            try:
                self.logger.info(f"Explore step {self.explore_step_num + 1} started.")
                explore_contionue = await self.explore_step()
                if not explore_contionue:
                    # TODO: support eval on last checkpoint
                    break
                if self.need_eval():
                    await self.eval()
                if await self.need_sync():
                    await self.sync_weight()
            except Exception:
                self.logger.error(f"Error in Explorer: {traceback.format_exc()}")
                break
        self.logger.info(
            f"--------------------\n> Explorer ({self.config.explorer.name}) finished.\n--------------------"
        )
        return self.config.explorer.name

    async def explore_step(self) -> bool:
        if self.explore_start_time is None:
            self.explore_start_time = time.time()
        try:
            tasks = await self.taskset.read_async()
        except StopAsyncIteration:
            self.logger.warning("No more tasks to explore. Stop exploring.")
            await self.save_checkpoint(sync_weight=False)
            await self.synchronizer.set_explorer_status.remote(
                RunningStatus.STOPPED,
                old_status=(
                    RunningStatus.RUNNING
                    if self.last_sync_successful
                    else RunningStatus.REQUIRE_SYNC
                ),
            )
            await self.shutdown()
            return False
        self.scheduler.schedule(tasks, batch_id=self.explore_step_num + 1)
        self.explore_step_num += 1
        return True

    async def need_sync(self) -> bool:
        if self.config.synchronizer.sync_style == SyncStyle.FIXED:
            if self.explore_step_num <= self.config.synchronizer.sync_offset:
                return False
            require_sync = (
                self.explore_step_num - self.config.synchronizer.sync_offset
            ) % self.config.synchronizer.sync_interval == 0
        else:
            require_sync = False
            if self.config.synchronizer.sync_style == SyncStyle.DYNAMIC_BY_EXPLORER:
                delta = self.explore_step_num - self.last_sync_step
                if delta >= self.config.synchronizer.sync_interval:
                    require_sync = True
            else:
                require_sync = await (
                    self.synchronizer.get_trainer_status.remote() == RunningStatus.REQUIRE_SYNC
                )
        if require_sync and self.last_sync_successful:
            await self.synchronizer.set_explorer_status.remote(
                RunningStatus.REQUIRE_SYNC, old_status=RunningStatus.RUNNING
            )
        return require_sync

    def need_eval(self) -> bool:
        return self.explore_step_num % self.config.explorer.eval_interval == 0

    async def eval(self):
        """Evaluation on all evaluation data samples."""
        self.eval_start_time = time.time()
        if len(self.config.buffer.explorer_input.eval_tasksets) == 0:
            self.logger.warning("No evaluation data samples. Skip evaluation.")
            return
        self.logger.info(f"Evaluation at step {self.explore_step_num} started.")

        if self.config.buffer.explorer_input.default_eval_workflow_type:
            self.logger.info(
                f"Use '{self.config.buffer.explorer_input.default_eval_workflow_type}' for evaluation."
            )

        for idx, eval_taskset_config in enumerate(self.config.buffer.explorer_input.eval_tasksets):
            self.logger.info(
                f"Evaluation on {eval_taskset_config.name} at step {self.explore_step_num} started."
            )
            # For eval tasksets, `total_steps` means max sampled tasks per eval run.
            max_eval_samples = eval_taskset_config.total_steps
            if max_eval_samples is not None and max_eval_samples <= 0:
                self.logger.warning(
                    f"Skip evaluation on {eval_taskset_config.name}: invalid total_steps={max_eval_samples}."
                )
                continue

            # Make random/shuffle eval sampling vary with eval step instead of repeating fixed subset.
            selector_type = eval_taskset_config.task_selector.selector_type
            if selector_type in {"random", "shuffle"}:
                eval_taskset_config.task_selector.seed = self.explore_step_num + idx

            eval_taskset = get_buffer_reader(eval_taskset_config)
            eval_batch_id = f"{self.explore_step_num}/{eval_taskset_config.name}"
            self.pending_eval_tasks.append((self.explore_step_num, eval_taskset_config.name))
            remaining_eval_samples = max_eval_samples
            while True:
                try:
                    read_batch_size = None
                    if remaining_eval_samples is not None:
                        if remaining_eval_samples <= 0:
                            break
                        read_batch_size = min(
                            max(1, eval_taskset_config.batch_size), remaining_eval_samples
                        )
                    data = await eval_taskset.read_async(batch_size=read_batch_size)
                    if remaining_eval_samples is not None:
                        remaining_eval_samples -= len(data)
                    self.scheduler.schedule(data, batch_id=eval_batch_id)
                except StopAsyncIteration:
                    break

    async def benchmark(self) -> bool:
        """Benchmark the model checkpoints."""
        # benchmark on the latest checkpoint
        if self.config.explorer.bench_on_latest_checkpoint:
            self.explore_step_num = await self._checkpoint_weights_update()
            await self.eval()
            await self._finish_eval_step(prefix="bench")
            return True

        # benchmark on base model
        if self.config.explorer.eval_on_startup:
            await self._finish_eval_step(prefix="bench")

        # benchmark on all checkpoints
        all_ckp_steps = sorted(
            [
                int(ckp.split("global_step_")[-1])
                for ckp in os.listdir(self.config.checkpoint_job_dir)
                if os.path.isdir(os.path.join(self.config.checkpoint_job_dir, ckp))
                and ckp.startswith("global_step_")
            ]
        )
        for step_num in all_ckp_steps:
            if step_num <= self.explore_step_num:
                continue
            self.explore_step_num = await self._checkpoint_weights_update(step_num=step_num)
            await self.eval()
            await self._finish_eval_step(prefix="bench")
        return True

    async def save_checkpoint(self, sync_weight: bool = False) -> None:
        if self.scheduler:
            if self.explore_step_num == 0:
                await self._finish_eval_step(step=0)
            else:
                await self._finish_steps(
                    self.last_monitored_step + 1, self.explore_step_num, self.model_version
                )
            self.last_monitored_step = self.explore_step_num

        if sync_weight:
            # sync weights
            self.logger.info(f"Explorer sync_weights at step {self.explore_step_num} started.")
            if self.use_nccl_sync:
                await self._nccl_weights_update()
            else:  # pull weights from Synchronizer
                await self._pull_latest_weights()
            self.logger.info(
                f"Explorer sync_weights at step {self.explore_step_num} finished, model version = {self.model_version}."
            )

        # save explore checkpoint
        self.state.save_explorer(
            current_step=self.explore_step_num,
            taskset_states=self.taskset.state_dict() if self.taskset else [],
        )

    async def sync_weight(self) -> None:
        """Synchronize model weights."""
        # call this method before training start to load the latest model weights
        await self.save_checkpoint(sync_weight=True)

    async def _finish_steps(self, start_step: int, end_step: int, model_version: int) -> None:
        for step in range(start_step, end_step + 1):
            self.logger.info(f"Waiting for step {step}")
            await self._finish_explore_step(step=step, model_version=model_version)
            await self._finish_eval_step(step=step)

        # Record the time: read_task + explore_step (>=1) + eval (if any)
        if self.explore_start_time is not None:
            metric = {"time/explorer_sync_interval": time.time() - self.explore_start_time}
            self.explore_start_time = None
            self.monitor.log(metric, step=end_step)

    async def _finish_explore_step(self, step: int, model_version: int) -> None:
        metric = {"rollout/model_version": model_version}
        if self.teacher_ema_metrics:
            metric.update(self.teacher_ema_metrics)
        with Timer(metric, "time/wait_explore_step"):
            statuses, exps = await self.scheduler.get_results(
                batch_id=step, min_num=self.min_wait_num
            )
        pipeline_metrics = await self.experience_pipeline.process.remote(exps)
        self.taskset.feedback(pipeline_metrics)
        metric.update(pipeline_metrics)
        if statuses:
            metric.update(gather_metrics([status.metrics[0] for status in statuses], "rollout"))
            metric["rollout/finished_task_count"] = len(statuses)
            self.monitor.log(metric, step=step)

    async def _finish_eval_step(self, step: Optional[int] = None, prefix: str = "eval") -> None:
        if not self.pending_eval_tasks:
            return
        step = step or self.explore_step_num
        metric = {}
        while self.pending_eval_tasks:
            eval_step, eval_task_name = self.pending_eval_tasks[0]
            if eval_step != step:
                return
            self.pending_eval_tasks.popleft()
            statuses, _ = await self.scheduler.get_results(batch_id=f"{step}/{eval_task_name}")
            metric[f"{prefix}/{eval_task_name}/finished_task_count"] = len(statuses)
            metric.update(
                gather_eval_metrics(
                    [status.metrics[0] for status in statuses],
                    f"{prefix}/{eval_task_name}",
                    detailed_stats=self.detailed_stats,
                )
            )
        if self.eval_start_time is not None:
            metric.update({"time/eval": time.time() - self.eval_start_time})
            self.eval_start_time = None
        self.monitor.log(metric, step)

    async def shutdown(self) -> None:
        if self.scheduler:
            await self.scheduler.stop()
            self.scheduler = None
        if self.experience_pipeline:
            await self.experience_pipeline.close.remote()
            # reserve `experience_pipeline.output` for trainer
            # TODO: refactor the lifecycle of buffer actor
            self._old_experience_pipeline = self.experience_pipeline
            self.experience_pipeline = None
        if self.monitor:
            self.monitor.close()
            self.monitor = None
        self.logger.info(
            f"Explorer ({self.config.explorer.name}) shutdown successfully at step {self.explore_step_num}."
        )

    async def is_alive(self) -> bool:
        """Check if the explorer is alive."""
        return True

    def _init_experience_pipeline(self) -> ray.actor.ActorHandle:
        """Init experience pipeline for the explorer."""
        if self.config.mode == "bench":
            return None
        node_id = ray.get_runtime_context().get_node_id()
        return (
            ray.remote(ExperiencePipeline)
            .options(
                name=f"{self.config.explorer.name}_pipeline",
                namespace=self.config.ray_namespace,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            )
            .remote(self.config)
        )

    @Experimental
    async def serve(self) -> None:
        """Run the explorer in serving mode.

        In serving mode, the explorer starts an OpenAI compatible server to handle requests.
        Agent applications can be deployed separately and interact with the explorer via the API.


        .. code-block:: python

            import openai


            client = openai.OpenAI(
                base_url=f"{explorer_server_url}/v1",
                api_key="EMPTY",
            )
            response = client.chat.completions.create(
                model=config.model.model_path,
                messages=[{"role": "user", "content": "Hello!"}]
            )
        """
        from trinity.explorer.proxy.service import ExplorerService

        self.service = ExplorerService(
            self,
            listen_address=self.config.explorer.listen_address,
            port=self.config.explorer.proxy_port,
        )
        await self.service.serve()
        self.server_url = f""
        self.logger.info(
            "======================================================\n"
            f"Starting Trinity Service on {self.server_url}\n"
            "======================================================"
        )
        self.state.save_explorer_server_url(self.server_url)
        while True:
            await asyncio.sleep(self.config.explorer.service_status_check_interval)
            # get the latest checkpoint
            model_version = await self.synchronizer.get_latest_model_version.remote()
            self.service.set_latest_model_version(model_version)

    @classmethod
    def get_actor(cls, config: Config):
        """Get a Ray actor for the explorer."""
        return (
            ray.remote(cls)
            .options(
                name=config.explorer.name,
                namespace=config.ray_namespace,
            )
            .remote(config)
        )
