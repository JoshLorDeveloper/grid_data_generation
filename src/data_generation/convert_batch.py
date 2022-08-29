import gym
import numpy as np
import os

import ray._private.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from .utils.constants import DAY_LENGTH

class BatchWriter:
    def __init__(self, out_path):
        self.batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
        self.writer = JsonWriter(out_path)
        self.step_data = {}
  
    def write_batch(self, episode_and_step, action, observation, reward):
        self.step_data[episode_and_step] = (action, observation, reward)
        prev_action, prev_observation, prev_reward = self.step_data.pop(
            (episode_and_step - 1), 
            (None, None, None),
        )
        if (prev_action is not None and
            prev_observation is not None and
            prev_reward is not None
        ):
            self.batch_builder.add_values(
                t=episode_and_step,
                eps_id=episode_and_step,
                agent_index=0,
                obs=prev_observation,
                actions=action,
                action_prob=1.0,  # put the true action probability here
                action_logp=0.0,
                rewards=reward,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=True,
                infos={},
                new_obs=observation
            )
            self.writer.write(self.batch_builder.build_and_reset())
  