# Systems For AI Project

## Requirements

* [PyTorch](https://pytorch.org/get-started/locally/)
* [OpenAI SpinngingUp](https://spinningup.openai.com/en/latest/user/installation.html)
* [RLLib](https://docs.ray.io/en/master/rllib.html)

## Reinforcement Learning Experiment Setup

Our experiments for testing ASP and EASGD against PPO, IMPALA and A3C are run on CloudLab. Below are the setup instructions.

To set up experiment on CloudLab:

1. Start a cluster on CloudLab
1. Run the following commands to install Docker on nodes:
    ```bash
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker <UNIQNAME>
    sudo systemctl restart docker -f
    ```
1. On worker nodes 1 and 2 (node 0 being the main node), run the following command to download and install Ray:
    ```bash
    sudo docker run -i -t --network host rayproject/ray-ml:nightly-cpu /bin/bash
    ```
1. Finally, run the following command (line 1 for node 1 and line 2 for node 2) in the ray terminal to connect the worker nodes to the main node:
    ```bash
    ray start --address='10.10.1.1:6379' --redis-password='5241590000000000' --node-ip-address='10.10.1.2'
    ray start --address='10.10.1.1:6379' --redis-password='5241590000000000' --node-ip-address='10.10.1.3'
    ```