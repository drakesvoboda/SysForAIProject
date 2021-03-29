import os

import logging
from typing import List, Tuple
import time
import collections
import copy

from collections import defaultdict

from ray.util.iter import from_actors, LocalIterator
from ray.util.iter_metrics import SharedMetrics
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, LEARNER_INFO, STEPS_TRAINED_COUNTER, \
    SAMPLE_TIMER, GRAD_WAIT_TIMER, _check_sample_batch_type, WORKER_UPDATE_TIMER, \
    _get_shared_metrics, LEARN_ON_BATCH_TIMER, LOAD_BATCH_TIMER, LAST_TARGET_UPDATE_TS
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch
from ray.rllib.utils.sgd import standardized, minibatches
from ray.rllib.utils.typing import PolicyID, SampleBatchType, ModelGradients
from ray.rllib.utils.sgd import do_minibatch_sgd, averaged

import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print

from ray.util.iter import from_actors, LocalIterator
from ray.rllib.agents import with_common_config
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches, \
    StandardizeFields, SelectExperiences
from ray.rllib.execution.train_ops import TrainOneStep, TrainTFMultiGPU
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.deprecation import DEPRECATED_VALUE
from ray.rllib.utils.typing import TrainerConfigDict
from ray.util.iter import LocalIterator
from ray.rllib.evaluation.rollout_worker import get_global_worker

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.execution.rollout_ops import AsyncGradients
from ray.rllib.execution.train_ops import ApplyGradients
from ray.rllib.execution.metric_ops import StandardMetricsReporting

import ray
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.ppo.ppo_tf_policy import ValueNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch, \
    Postprocessing
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.tf_policy import LearningRateSchedule
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_ops import explained_variance

from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, LEARNER_INFO, \
    SAMPLE_TIMER, GRAD_WAIT_TIMER, _check_sample_batch_type, \
    _get_shared_metrics, _get_global_vars

from ray.rllib.utils.typing import PolicyID, SampleBatchType, ModelGradients

from asp.asp_tf_policy import ASPTFPolicy

DEFAULT_CONFIG = with_common_config({
    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # Size of rollout batch
    "rollout_fragment_length": 10,
    # GAE(gamma) parameter
    "lambda": 1.0,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Learning rate
    "lr": 0.0001,
    # Learning rate schedule
    "lr_schedule": None,
    # Value Function Loss coefficient
    "vf_loss_coeff": 0.5,
    # Entropy coefficient
    "entropy_coeff": 0.01,
    # Min time per iteration
    "min_iter_time_s": 5,
    # Workers sample async. Note that this increases the effective
    # rollout_fragment_length by up to 5x due to async buffering of batches.
    "sample_async": True,

    "significance_threshold": 0
})

class ASPUpdateLearnerWeights:
    def __init__(self, workers, num_workers, significance_threshold):
        self.total_steps = 0
        self.workers = workers

        self.counters = [i for i in range(num_workers)]
        self.significance_threshold = significance_threshold

        self.counters = {actor: 0 for actor in self.workers.remote_workers()}
        self.worker_idx = {actor: idx for idx, actor in enumerate(self.workers.remote_workers())}

    def __call__(self, item):
        actor, (info, samples, training_steps) = item

        lw = self.workers.local_worker()

        metrics = _get_shared_metrics()

        metrics.counters[STEPS_TRAINED_COUNTER] += training_steps
        metrics.counters[STEPS_SAMPLED_COUNTER] += samples

        self.counters[actor] += 1

        metrics.counters[f"WorkerIteration/Worker{self.worker_idx[actor]}"] += 1

        global_vars = _get_global_vars()
        lw.set_global_vars(global_vars)
        actor.set_global_vars.remote(global_vars)

        with metrics.timers[WORKER_UPDATE_TIMER]:
            for pid, p in lw.policy_map.items():
                def get_update(w, significance_threshold):
                    return w.policy_map[pid].asp_get_updates(significance_threshold)

                def sync_update(w, update):
                    return w.policy_map[pid].asp_sync_updates(update)
                
                update = actor.apply.remote(get_update, self.significance_threshold)

                if lw != actor: sync_update(lw, ray.get(update))

                if self.workers.remote_workers():
                    for e in self.workers.remote_workers():
                        if e != actor: e.apply.remote(sync_update, update)

        return info

def LocalTrainOneStep(workers: WorkerSet, num_sgd_iter: int = 1, sgd_minibatch_size: int = 0):
    rollouts = from_actors(workers.remote_workers())

    def train_on_batch(samples):
        if isinstance(samples, SampleBatch):
            samples = MultiAgentBatch({DEFAULT_POLICY_ID: samples}, samples.count)

        worker = get_global_worker()

        if not hasattr(worker, 'num_iterations_trained'):
            worker.num_iterations_trained = 0

        if num_sgd_iter > 1:
            info = do_minibatch_sgd(samples, {pid: worker.get_policy(pid) for pid in worker.policies_to_train}, worker, num_sgd_iter, sgd_minibatch_size, [])
        else:
            info = worker.learn_on_batch(samples)

        worker.num_iterations_trained += 1
        info['num_iterations_trained'] = worker.num_iterations_trained

        return info, samples.count, num_sgd_iter

    info = rollouts.for_each(train_on_batch)

    return info

def asp_execution_plan(workers, config):
    workers.sync_weights()

    workers.foreach_trainable_policy(lambda p, pid: p.asp_sync_global_model())


    if "num_sgd_iter" in config:
        train_op = LocalTrainOneStep(workers, num_sgd_iter=config["num_sgd_iter"], sgd_minibatch_size=config["sgd_minibatch_size"])
    else:
        train_op = LocalTrainOneStep(workers)

    if workers.remote_workers():
        train_op = train_op.gather_async().zip_with_source_actor() \
            .for_each(ASPUpdateLearnerWeights(workers, config['num_workers'], config['significance_threshold']))

    return StandardMetricsReporting(train_op, workers, config)

def get_policy_class(config):
    return ASPTFPolicy

def validate_config(config):
    if config["entropy_coeff"] < 0:
        raise ValueError("`entropy_coeff` must be >= 0.0!")
    if config["num_workers"] <= 0 and config["sample_async"]:
        raise ValueError("`num_workers` for A3C must be >= 1!")

ASPTrainer = a3c.A3CTrainer.with_updates(
    name="EASGD",
    default_policy=ASPTFPolicy,
    default_config=DEFAULT_CONFIG,
    get_policy_class=get_policy_class,
    execution_plan=asp_execution_plan)