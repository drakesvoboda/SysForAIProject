"""Note: Keep in sync with changes to VTraceTFPolicy."""
import collections

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

tf1, tf, tfv = try_import_tf()

class EASGDUpdate:
    @staticmethod
    def diff(local_param, global_param):
        if isinstance(global_param, collections.MutableMapping):
            return { key: EASGDUpdate.diff(local, glob) for (key, local), (_, glob) in zip(local_param.items(), global_param.items()) }
        else:
            return local_param - global_param

    @staticmethod
    def easgd_add(params, diffs, alpha):
        if isinstance(params, collections.MutableMapping):
            return { key: EASGDUpdate.easgd_add(param, diff, alpha) for (key, param), (_, diff) in zip(params.items(), diffs.items()) }
        else:
            return params + alpha * diffs

    @staticmethod
    def easgd_subtract(params, diffs, alpha):
        if isinstance(params, collections.MutableMapping):
            return { key: EASGDUpdate.easgd_subtract(param, diff, alpha) for (key, param), (_, diff) in zip(params.items(), diffs.items()) }
        else:
            return params - alpha * diffs

class EASGDUpdateMixin:
    def __init__(self): 
        def do_update(global_weights, alpha):
            local_weights = self.get_weights()
            diff = EASGDUpdate.diff(local_weights, global_weights)

            self.set_weights(EASGDUpdate.easgd_subtract(local_weights, diff, alpha))

            return diff

        self.easgd_update = do_update  

def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    EASGDUpdateMixin.__init__(policy)


def get_default_config():
    from easgd import DEFAULT_CONFIG
    return DEFAULT_CONFIG

EASGDTFPolicy = A3CTFPolicy.with_updates(
    name="EASGDTFPolicy",
    get_default_config=get_default_config,
    mixins=[ValueNetworkMixin, LearningRateSchedule, EASGDUpdateMixin], 
    before_loss_init=setup_mixins,
)
