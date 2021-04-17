import time
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.impala as impala
import asp
import easgd
from ray.tune.logger import pretty_print

ray.init()
runtime = 3600
# num_workers = 5
num_envs_per_worker = 5
# envs = ["Acrobot-v1","CartPole-v1","Blackjack-v0"]
envs = ["Blackjack-v0"]

def run_policy(trainer):
   start = int(round(time.time()))
   while True:
      elapsed = int(round(time.time())) - start
      if elapsed > runtime:
         break
      result = trainer.train()
      print(pretty_print(result))

for env in envs:
   for num_workers in [5,10]:
      config = ppo.DEFAULT_CONFIG.copy()
      config["num_gpus"] = 0
      config["num_workers"] = num_workers
      config["num_envs_per_worker"] = num_envs_per_worker
      trainer = ppo.PPOTrainer(config=config, env=env)
      run_policy(trainer)
      
      config = a3c.DEFAULT_CONFIG.copy()
      config["num_gpus"] = 0
      config["num_workers"] = num_workers
      config["num_envs_per_worker"] = num_envs_per_worker
      trainer = a3c.A3CTrainer(config=config, env=env)
      run_policy(trainer)
      
      config = impala.DEFAULT_CONFIG.copy()
      config["num_gpus"] = 0
      config["num_workers"] = num_workers
      config["num_envs_per_worker"] = num_envs_per_worker
      trainer = impala.ImpalaTrainer(config=config, env=env)
      run_policy(trainer)
      
      config = asp.DEFAULT_CONFIG.copy()
      config["num_gpus"] = 0
      config["num_workers"] = num_workers
      config["num_envs_per_worker"] = num_envs_per_worker
      config["lr_schedule"] = [[0, 0.0007],[20000000, 0.000000000001],]
      config["significance_threshold"] = 0.1
      trainer = asp.ASPTrainer(config=config, env=env)
      run_policy(trainer)
      
      config = easgd.DEFAULT_CONFIG.copy()
      config["num_gpus"] = 0
      config["num_workers"] = num_workers
      config["num_envs_per_worker"] = num_envs_per_worker
      config["lr_schedule"] = [[0, 0.0007],[20000000, 0.000000000001],]
      config["moving_rate"] = 0.9
      config["update_frequency"] = 5
      trainer = easgd.EASGDTrainer(config=config, env=env)
      run_policy(trainer)