""" Custom implementation of proximal policy optimization. Multiple parallel actors simulate the environment and 
run local copies of the policy. Experiences are gathered by a learner processes that performs SGD to update the policy.
The new policy is then sent back to the actor processes. """

import time
import random
from functools import reduce
import numpy as np
from mpi4py import MPI
from tqdm import tqdm, trange

import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler

import gym
from gym.spaces import Box, Discrete
import spinup.algos.pytorch.ppo.core as core
import queue

from ActorCritic import MLPActorCritic

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class TrainingQueue():
    def __init__(self, max_num):
        self.max_num = max_num
        self.queue = []

        self.adv_mean = None
        self.adv_std = None

    def put_batch(self, batch):
        for item in batch:
            self.queue.append(item)

        while len(self.queue) > self.max_num:
            self.queue.pop(0)

        adv = torch.tensor([d['adv'] for d in self.queue])

        # self.adv_mean = adv.mean()
        # self.adv_std = adv.std()

    def get_batch(self, n):
        """ Pop a batch of n items from the queue """

        if len(self.queue) <= 0:
            return {"obs": torch.tensor([]), "act": torch.tensor([]), "ret": torch.tensor([]), "adv": torch.tensor([]), "logp": torch.tensor([])}

        result = random.sample(self.queue, n)

        try:
            result = {k: torch.tensor([d[k] for d in result]) for k in result[0]}
        except:
            print(result[0])
            print({k: [d[k] for d in result] for k in result[0]})
            raise Exception()


        # result['adv'] = (result['adv'] - self.adv_mean) / self.adv_std
        
        return result

def actor_loss(adv, logp, logp_old, clip_ratio): 
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # Useful extra info
    approx_kl = (logp_old - logp).mean().item()
    clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
    clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
    pi_info = dict(kl=approx_kl, cf=clipfrac)

    return loss_pi, pi_info

def critic_loss(pred, ret):  
    return ((pred - ret)**2).mean()

def run_update(logp, pred_return, ret, adv, logp_old, pi_optimizer, vf_optimizer, clip_ratio, logger):      
    vf_loss = critic_loss(pred_return, ret)
    vf_optimizer.zero_grad()
    vf_loss.backward()
    vf_optimizer.step()

    pi_loss, pi_info = actor_loss(adv, logp, logp_old, clip_ratio)
    pi_optimizer.zero_grad()
    pi_loss.backward()
    pi_optimizer.step()

   
class PPOTrajectory:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma, self.lam = gamma, lam
        self.reset()
    
    def store(self, obs, act, rew, val, logp):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.val_buf.append(val)
        self.logp_buf.append(logp)

    def reset(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.val_buf = []
        self.logp_buf = []

    def finish_path(self, last_val=0):
        self.rew_buf.append(last_val)
        self.val_buf.append(last_val)

        rews = np.array(self.rew_buf)
        vals = np.array(self.val_buf)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv_buf = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        ret_buf = core.discount_cumsum(rews, self.gamma)[:-1]

        trajectory = [dict(obs=obs, act=act, ret=ret, adv=adv, logp=logp) \
            for obs, act, ret, adv, logp in zip(self.obs_buf, self.act_buf, ret_buf, adv_buf, self.logp_buf)]

        self.reset()

        """ TODO: Advantage normalization """

        return trajectory

def ppo(env_fn, actor_critic=MLPActorCritic, ac_kwargs={}, seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    torch.manual_seed(10)
    np.random.seed(10)

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Instantiate environment
    env = env_fn()

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs).cuda()

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOTrajectory(gamma, lam)
    training_queue = TrainingQueue(256)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    num_training = 7

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):

            training_data = training_queue.get_batch(num_training)
            num_training = len(training_data['act'])

            o = torch.tensor(o).float().unsqueeze(0)

            if num_training > 0:
                o = torch.cat([o, training_data['obs']])

            o = o.cuda()

            pi = ac.pi._distribution(o)
            a = pi.sample()

            if num_training > 0:
                a[-num_training:] = training_data['act']

            logp = ac.pi._log_prob_from_distribution(pi, a)
            v = ac.v(o)

            if num_training > 0:
                run_update(logp[-num_training:], 
                            v[-num_training:], 
                            training_data['ret'].cuda(), 
                            training_data['adv'].cuda(), 
                            training_data['logp'].cuda(), 
                            pi_optimizer, 
                            vf_optimizer, 
                            clip_ratio, 
                            logger)

            a = a[:len(a)-num_training].cpu().item()
            o = o[:len(o)-num_training].cpu().numpy().squeeze()
            v = v[:len(v)-num_training].cpu().item()
            logp = logp[:len(logp)-num_training].cpu().item()

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, logp)
            logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o
            num_training = 7

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch-1

            if terminal:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' %
                        ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    v = ac.v(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).cuda()).cpu().detach().item()
                else:
                    v = 0
                trajectory = buf.finish_path(v)
                training_queue.put_batch(trajectory)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        # logger.log_tabular('ClipFrac', average_only=True)
        # logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs('PPO', args.seed)

    from stable_baselines.common import make_vec_env

    ppo(lambda : gym.make('CartPole-v0'), actor_critic=MLPActorCritic, gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, pi_lr=1e-3, vf_lr=1e-3, target_kl=0.05, train_pi_iters=80, train_v_iters=80)