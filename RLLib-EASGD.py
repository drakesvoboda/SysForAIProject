import os

# os.environ["TUNE_RESULT_DIR"] = "/media/drake/BlackPassport/ray_results/"


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

ray.init(address='10.10.1.1:6379')

config = a3c.DEFAULT_CONFIG.copy()
# config = ppo.DEFAULT_CONFIG.copy()

config["num_gpus"] = 0
config["num_workers"] = 25
config["sample_async"] = False
config["num_envs_per_worker"] = 5
# config["rollout_fragment_length"] = 100
config["lr_schedule"] = [[0, 0.0007],[20000000, 0.000000000001],]

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
        actor, (info, samples, training_steps) = item

        metrics = _get_shared_metrics()

        metrics.counters[STEPS_TRAINED_COUNTER] += training_steps
        metrics.counters[STEPS_SAMPLED_COUNTER] += samples

        self.counters[actor] += 1

        if self.counters[actor] % self.broadcast_interval == 0:
            
            metrics.counters["num_weight_broadcasts"] += 1

            with metrics.timers[WORKER_UPDATE_TIMER]:

                local_weights = ray.get(actor.get_weights.remote())

                diff_dict = EASGDUpdateLearnerWeights.diff(local_weights, self.global_weights)
                
                # print(actor)
                # print(diff_dict["default_policy"]["default_policy/value_out/kernel"][:10])
                #print(local_weights["default_policy"]["default_policy/value_out/kernel"][:10])
                #print(self.global_weights["default_policy"]["default_policy/value_out/kernel"][:10])

                self.global_weights = self.easgd_add(self.global_weights, diff_dict)
                local_weights = self.easgd_subtract(local_weights, diff_dict)
                    
                # Update metrics.    
                actor.set_weights.remote(local_weights, _get_global_vars())

                # Also update global vars of the local worker.
                # self.workers.local_worker().set_global_vars(_get_global_vars())

        return info

def log_weights(items):
    actor, items = items

    weights = ray.get(actor.get_weights.remote())
    print(weights["default_policy"]["default_policy/value_out/kernel"][0:10])

def LocalTrainOneStepV0(workers: WorkerSet, num_sgd_iter: int = 1, sgd_minibatch_size: int = 0):
    workers.sync_weights()

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

def LocalTrainOneStep(workers: WorkerSet, num_sgd_iter: int = 1, sgd_minibatch_size: int = 0):
    workers.sync_weights()

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

def easgd_execution_plan(workers, config):
    if "num_sgd_iter" in config:
        train_op = LocalTrainOneStepV0(workers, num_sgd_iter=config["num_sgd_iter"], sgd_minibatch_size=config["sgd_minibatch_size"])
    else:
        train_op = LocalTrainOneStep(workers)

    if workers.remote_workers():
        train_op = train_op.gather_async().zip_with_source_actor().for_each(EASGDUpdateLearnerWeights(workers, .9, config["num_workers"], 20))

    return StandardMetricsReporting(train_op, workers, config)

#CustomTrainer = PPOTrainer.with_updates(
#    name="EASGD",
#    execution_plan=easgd_execution_plan)

CustomTrainer = a3c.A3CTrainer.with_updates(
    name="EASGD-A3C",
    execution_plan=easgd_execution_plan)

# CustomTrainer = a3c.A3CTrainer.with_updates()

trainer = CustomTrainer(config=config, env="QbertNoFrameskip-v4")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   # print("==========")
   print(pretty_print(result))