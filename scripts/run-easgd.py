import os

# os.environ["TUNE_RESULT_DIR"] = "/media/drake/BlackPassport/ray_results/"

import ray
import easgd
from ray.tune.logger import pretty_print

ray.init(address='10.10.1.1:6379')

config = easgd.DEFAULT_CONFIG.copy()

config["num_gpus"] = 0
config["num_workers"] = 10
config["num_envs_per_worker"] = 5
config["lr_schedule"] = [[0, 0.0007],[20000000, 0.000000000001],]
config["moving_rate"] = .9
config["update_frequency"] = 20

trainer = easgd.EASGDTrainer(config=config, env="QbertNoFrameskip-v4")

for i in range(1000):
   result = trainer.train()
   print(pretty_print(result))