from gym.envs.mujoco import HalfCheetahEnv

from garage.envs import GarageEnv
from garage.envs.multi_task_metaworld_wrapper import MTMetaWorldWrapper
from garage.envs import normalize_reward
from garage.envs import ML1WithPinnedGoal

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import dowel
from dowel import logger as dowel_logger
from dowel import tabular as dowel_tab

from datetime import datetime
import os
import pickle

now = datetime.now() # current date and time
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
experiment_name = 'push_v1_' + str(date_time)
dowel_logger.add_output(dowel.StdOutput())
dowel_logger.add_output(dowel.CsvOutput(os.path.join(experiment_name, 'progress.csv')))
dowel_logger.add_output(dowel.TensorBoardOutput(experiment_name, x_axis='TotalEnvSteps'))

def get_ML1_envs(name):
    bench = ML1WithPinnedGoal.get_train_tasks(name)
    tasks = [{'task': 0, 'goal': i} for i in range(50)]
    ret = {}
    for task in tasks:
        new_bench = bench.clone(bench)
        new_bench.set_task(task)
        ret[("goal"+str(task['goal']))] = GarageEnv(normalize_reward(new_bench.active_env))
    return ret

def get_ML1_envs_test(name):
    bench = ML1WithPinnedGoal.get_train_tasks(name)
    tasks = [{'task': 0, 'goal': i} for i in range(50)]
    ret = {}
    for task in tasks:
        new_bench = bench.clone(bench)
        new_bench.set_task(task)
        ret[("goal"+str(task['goal']))] = GarageEnv((new_bench.active_env))
    return ret


def experiment(variant):
    expl_env =  MTMetaWorldWrapper(get_ML1_envs("push-v1"))
    eval_env = pickle.loads(pickle.dumps(expl_env))
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=400,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=500,
            num_eval_steps_per_epoch=7500*5,
            num_trains_per_train_loop=150,
            num_train_loops_per_epoch=26,
            num_expl_steps_per_train_loop=7500,
            min_num_steps_before_training=7500,
            max_path_length=150,
            batch_size=1280,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    # if args.log_dir is None:
    #     log_dir = os.path.join(os.path.join(os.getcwd(), 'data'),
    #                            experiment_name)
    setup_logger(experiment_name, variant=variant, snapshot_mode='none')
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
