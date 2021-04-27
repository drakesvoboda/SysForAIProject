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
   1. ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker <UNIQNAME>
   sudo systemctl restart docker -f
   ```
   1. Item 3b