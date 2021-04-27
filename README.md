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

To set up local files:

1. Run `pip install -U ray` to install Ray on your local machine.
1. Update line 39 of `cluster.yaml` and set the `head_ip` to the IP address of node 0 (which can be found using the `ifconfig` command)
1. Update line 49 of `cluster.yaml` and set the `ssh_user` to your UNIQNAME
1. Update line 51 of `cluster.yaml` and set the `ssh_private_key` to your key file used to set up the CloudLab account

To set up master node and start experiments:

1. To start Ray and initially install Ray on the master node, run:
    ```bash
    ray up -y cluster.yaml
    ```
1. To test the installation and setup, run:
    ```bash
    ray submit cluster.yaml scripts/cluster-test.py
    ```
1. To submit a script to run on Ray, run:
    ```bash
    ray submit cluster.yaml <filename_path>
    ```

These and many other useful commands can be found in the file `useful-commands.py`.

Our implementations for ASP and EASGD can be found in the `asp` and `easgd` folders.

The files executed on Ray can be found in the scripts folder. The `cluster-run.py` file runs multiple experiments and contains the settings for many different environments.