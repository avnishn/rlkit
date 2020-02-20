import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import numpy as np
import dowel
from dowel import logger as dowel_logger
from dowel import tabular as dowel_tab

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.total_env_steps = 0

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            self.total_env_steps += self.min_num_steps_before_training

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            import time
            t = time.time()
            new_eval_paths = self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )

            success_on_paths = []
            for eval_path in new_eval_paths:
                current_trajectory_success_rates = []
                for step_wise_env_info in eval_path['env_infos']:
                    current_trajectory_success_rates.append(step_wise_env_info["success"])
                success = float(np.array(current_trajectory_success_rates).any())
                success_on_paths.append(success)
            success_rate = np.mean(success_on_paths)

            # print(time.time()-t)
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                t = time.time()
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                self.total_env_steps += self.num_expl_steps_per_train_loop
                # print(time.time()-t)
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                t =time.time()
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                # print(time.time()-t)
                gt.stamp('training', unique=False)
                self.training_mode(False)
            dowel_tab.record("TotalEnvSteps", self.total_env_steps)
            dowel_tab.record("SuccessRate", success_rate)
            dowel_logger.log(dowel_tab)
            dowel_logger.dump_all()
            self._end_epoch(epoch)
