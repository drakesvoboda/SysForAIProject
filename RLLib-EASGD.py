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

from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, LEARNER_INFO, \
    SAMPLE_TIMER, GRAD_WAIT_TIMER, _check_sample_batch_type, \
    _get_shared_metrics, _get_global_vars

from ray.rllib.utils.typing import PolicyID, SampleBatchType, ModelGradients

ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 4
config["num_envs_per_worker"] = 4
config["rollout_fragment_length"] = 200


def a3c_execution_plan(workers, config):
    # For A3C, compute policy gradients remotely on the rollout workers.
    grads = AsyncGradients(workers)

    # Apply the gradients as they arrive. We set update_all to False so that
    # only the worker sending the gradient is updated with new weights.
    train_op = grads.for_each(ApplyGradients(workers, update_all=False))

    return StandardMetricsReporting(train_op, workers, config)

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class EASGDUpdateLearnerWeights:
    def __init__(self, workers, moving_rate, num_workers, broadcast_interval):
        self.total_steps = 0
        self.broadcast_interval = broadcast_interval
        self.workers = workers
        self.global_weights = copy.deepcopy(workers.local_worker().get_weights())
        self.counters = [i for i in range(num_workers)]
        self.alpha = moving_rate / num_workers

        self.counters = {actor: 0 for actor in self.workers.remote_workers()}

    @staticmethod
    def diff(local_param, global_param):
        if isinstance(global_param, collections.MutableMapping):
            return { key: EASGDUpdateLearnerWeights.diff(local, glob) for (key, local), (_, glob) in zip(local_param.items(), global_param.items()) }
        else:
            return local_param - global_param

    def easgd_add(self, params, diffs):
        if isinstance(params, collections.MutableMapping):
            return { key: self.easgd_add(param, diff) for (key, param), (_, diff) in zip(params.items(), diffs.items()) }
        else:
            return params + self.alpha * diffs

    def easgd_subtract(self, params, diffs):
        if isinstance(params, collections.MutableMapping):
            return { key: self.easgd_subtract(param, diff) for (key, param), (_, diff) in zip(params.items(), diffs.items()) }
        else:
            return params - self.alpha * diffs

    def __call__(self, item):
        actor, batch = item

        self.counters[actor] += 1

        if self.counters[actor] % self.broadcast_interval == 0:
            metrics = _get_shared_metrics()
            metrics.counters["num_weight_broadcasts"] += 1

            with metrics.timers[WORKER_UPDATE_TIMER]:

                local_weights = ray.get(actor.get_weights.remote())

                diff_dict = EASGDUpdateLearnerWeights.diff(local_weights, self.global_weights)
                
                #print(worker_idx)
                #print(diff_dict["default_policy"]["default_policy/value_out/kernel"][:10])
                #print(local_weights["default_policy"]["default_policy/value_out/kernel"][:10])
                #print(self.global_weights["default_policy"]["default_policy/value_out/kernel"][:10])

                self.global_weights = self.easgd_add(self.global_weights, diff_dict)
                local_weights = self.easgd_subtract(local_weights, diff_dict)
                    
                # Update metrics.    
                actor.set_weights.remote(local_weights, _get_global_vars())

                # Also update global vars of the local worker.
                # self.workers.local_worker().set_global_vars(_get_global_vars())


def log_weights(items):
    actor, items = items

    weights = ray.get(actor.get_weights.remote())
    print(weights["default_policy"]["default_policy/value_out/kernel"][0:10])


def LocalTrainOneStep(workers: WorkerSet, num_sgd_iter: int = 1, sgd_minibatch_size: int = 0):
    rollouts = from_actors(workers.remote_workers())

    def train_on_batch(samples):
        if isinstance(samples, SampleBatch):
            samples = MultiAgentBatch({DEFAULT_POLICY_ID: samples}, samples.count)

        worker = get_global_worker()

        def train_policy(policy, policy_id):
            batch = samples.policy_batches[policy_id]

            for i in range(num_sgd_iter):
                for minibatch in minibatches(batch, sgd_minibatch_size):
                    batch_fetches = (worker.learn_on_batch(
                        MultiAgentBatch({
                            policy_id: minibatch
                        }, minibatch.count)))[policy_id]
        
        worker.foreach_trainable_policy(train_policy)

        return {}

    info = rollouts.for_each(train_on_batch)

    return info


def easgd_execution_plan(workers, config):
    train_op = LocalTrainOneStep(workers, num_sgd_iter=config["num_sgd_iter"], sgd_minibatch_size=config["sgd_minibatch_size"])

    if workers.remote_workers():
        train_op = train_op.gather_async().zip_with_source_actor().for_each(EASGDUpdateLearnerWeights(workers, 1, config["num_workers"], 1))

    return StandardMetricsReporting(train_op, workers, config)

CustomTrainer = PPOTrainer.with_updates(
    execution_plan=easgd_execution_plan)

trainer = CustomTrainer(config=config, env="CartPole-v0")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   # print("==========")
   print(pretty_print(result))