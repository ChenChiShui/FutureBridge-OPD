#!/usr/bin/env bash
set -euo pipefail

: "${GIGPO_DIR:?Set GIGPO_DIR to the GiGPO checkout}"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-8B model}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the prepared training parquet}"
: "${VAL_DATA:?Set VAL_DATA to the prepared validation parquet}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for the Teacher outputs}"

WEBSHOP_DIR=${WEBSHOP_DIR:-"${GIGPO_DIR}/agent_system/environments/env_package/webshop/webshop"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-webshop_gigpo_teacher}

export PYTHONPATH="${GIGPO_DIR}:${WEBSHOP_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "${GIGPO_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    "data.train_files=${TRAIN_DATA}" \
    "data.val_files=${VAL_DATA}" \
    data.train_batch_size=16 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    env.env_name=Webshop \
    env.max_steps=15 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.25 \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.project_name=webshop_gigpo_teacher \
    "trainer.experiment_name=${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    "trainer.default_local_dir=${OUTPUT_DIR}" \
    trainer.val_before_train=True
