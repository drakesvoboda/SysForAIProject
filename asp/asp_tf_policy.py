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

import numpy as np

tf1, tf, tfv = try_import_tf()

class ASPUpdate:
    @staticmethod
    def diff(local_param, global_param):
        if isinstance(global_param, collections.MutableMapping):
            return { key: ASPUpdate.diff(local, glob) for (key, local), (_, glob) in zip(local_param.items(), global_param.items()) }
        else:
            return local_param - global_param

    @staticmethod
    def subtract(params, diffs=None):
        if diffs is None or len(diffs) == 0: 
            return params
        elif isinstance(params, collections.MutableMapping):
            return { key: ASPUpdate.subtract(param, diffs[key] if key in diffs else None) for key, param in params.items() }
        else:
            return params - diffs

    @staticmethod
    def add(params, diffs=None, scale_factor=1):
        if diffs is None or len(diffs) == 0: 
            return params
        elif isinstance(params, collections.MutableMapping):
            return { key: ASPUpdate.add(param, diffs[key] if key in diffs else None, scale_factor) for key, param in params.items() }
        else:
            return params + diffs*scale_factor

    def significance_filter(params, diffs, threshold, num_significant=0):
        def is_significant(param, diff):
            if isinstance(diff, collections.MutableMapping): return True

            norm_params = np.linalg.norm(param)
            if norm_params == 0: return True

            return np.linalg.norm(diff)/norm_params > threshold



        if isinstance(params, collections.MutableMapping):
            ret = {}

            for (key, param), (_, diff) in zip(params.items(), diffs.items()):
                if is_significant(param, diff):
                    children, num_children = ASPUpdate.significance_filter(param, diff, threshold)
                    
                    if len(children) > 0:
                        ret[key] = children

                    num_significant += num_children
            
            return ret, num_significant
        else:
            return diffs, 1
            
class ASPUpdateMixin:
    def __init__(self): 
        def sync_global_model():
            # I'm using the global weights parameter to indirectly compute the accumulated updates. It might be better to do that more directly.
            self.global_weights = self.get_weights()

        def do_update(update):
            lw = self.get_weights()
            lw = ASPUpdate.add(lw, update)
            self.set_weights(lw)
            self.global_weights = ASPUpdate.add(self.global_weights, update)

        def get_updates(significance_threshold):
            local_weights = self.get_weights()
            update = ASPUpdate.diff(local_weights, self.global_weights)
            update, num_significant = ASPUpdate.significance_filter(local_weights, update, significance_threshold)
            self.global_weights = ASPUpdate.add(self.global_weights, update)
            return update, num_significant

        self.asp_sync_global_model = sync_global_model
        self.asp_sync_updates = do_update
        self.asp_get_updates = get_updates  

def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    ASPUpdateMixin.__init__(policy)

def get_default_config():
    from asp import DEFAULT_CONFIG
    return DEFAULT_CONFIG

ASPTFPolicy = A3CTFPolicy.with_updates(
    name="ASPTFPolicy",
    get_default_config=get_default_config,
    mixins=[ValueNetworkMixin, LearningRateSchedule, ASPUpdateMixin], 
    before_loss_init=setup_mixins,
)
