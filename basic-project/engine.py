import ray
from pathlib import Path
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved

from commons.methods import load_yaml

class Engine:
    """
    Interface class between ray and farma gym

    Args:
        :param [config]: configuration dictionary for engine

    Internal State:
        :param [trainable]: algorithm and objective function to be trained on the problem
        :param [tune_config]: tuner configuration such as HPO
        :param [param_space_config]: algorithm hyperparameter space configuration,
            framework, environment configuration and rollout policy
        :param [ai_runtime_config]: AI runtine configuration such as stop conditions and chkpts
    """

    def __init__(self, config: dict):
        self.config = config

        self._trainable = None
        self._param_space_config = None
        self._ai_runtime_config = None

    def _set_trainable(self, trainable):
        """ set trainable algorithm for the objective """
        self._trainable = trainable["choice"]

    def _set_param_space_config(self, param_space_configuration):
        """ fill the configuration with needed parameters """
        self._param_space_config = (
            PPOConfig()
            .environment(param_space_configuration["environment"]["env_name"]["choice"])
            .framework(param_space_configuration["framework"]["choice"])
            .rollouts(
                num_rollout_workers=param_space_configuration["rollouts"]["num-workers"]["choice"],
                num_envs_per_worker=param_space_configuration["rollouts"]["num-envs-per-worker"]["choice"],
                rollout_fragment_length=param_space_configuration["rollouts"]["rollout_fragment_length"]["choice"],
            )
            .training(
                lr=param_space_configuration["training"]["lr"]["choice"],
                lambda_=param_space_configuration["training"]["lambda_"]["choice"],
                gamma=param_space_configuration["training"]["gamma"]["choice"],
                sgd_minibatch_size=param_space_configuration["training"]["sgd_minibatch_size"]["choice"],
                use_gae=param_space_configuration["training"]["use_gae"]["choice"],
                train_batch_size=param_space_configuration["training"]["train_batch_size"]["choice"],
                num_sgd_iter=param_space_configuration["training"]["num_sgd_iter"]["choice"],
                clip_param=param_space_configuration["training"]["clip_param"]["choice"],
                model=param_space_configuration["training"]["model"]["choice"],
            )
            .resources(num_gpus=param_space_configuration["resources"]["num_gpus"]["choice"])
            .evaluation(
                evaluation_interval=param_space_configuration["evaluation"]["evaluation-interval"]["choice"],
                evaluation_duration=param_space_configuration["evaluation"]["evaluation-duration"]["choice"],
            )
        )

    def _set_ai_runtime_config(self, run_config, stop_config, checkpoint_config):
        """ set configuration for AI runtime configuration"""
        stop_conditions = {
            "training_iteration": stop_config["training_iteration"]["choice"],
            "timesteps_total": stop_config["timesteps_total"]["choice"],
            "episode_reward_mean": stop_config["episode_reward_mean"]["choice"],
        }
        checkpoint_conditions = air.CheckpointConfig(
            checkpoint_frequency=checkpoint_config["checkpoint_frequency"]["choice"],
            checkpoint_at_end=checkpoint_config["checkpoint_at_end"]["choice"]
        )

        verbose = run_config["verbose"]["choice"]
        self._ai_runtime_config = air.RunConfig(
            stop=stop_conditions,
            checkpoint_config=checkpoint_conditions,
            verbose=verbose,
        )

    def _instantiate_tuner_entity(self):
        tuner = tune.Tuner(
            self._trainable,
            param_space=self._param_space_config.to_dict(),
            run_config=self._ai_runtime_config,
        )
        return tuner

    def load_configuration_and_set_parameters(self):
        """ fetch given configuration and inject it into the internal parameters space """
        param_space = self.config["param-space"]
        tuner_space = self.config["tuner-space"]

        ai_runtime_conditions = tuner_space["ai-runtime-conditions"]
        trainable = tuner_space["trainable"]

        run_config = ai_runtime_conditions["run-config"]
        stop_config = ai_runtime_conditions["stop-config"]
        checkpoint_config = ai_runtime_conditions["checkpoint-config"]

        self._set_trainable(trainable)
        self._set_param_space_config(param_space)
        self._set_ai_runtime_config(run_config, stop_config, checkpoint_config)

    def start_rollout_and_learn(self):
        """ start ray cluster and learn the objective """
        ray.init()
        tuner = self._instantiate_tuner_entity()
        results = tuner.fit()

        ray.shutdown()
        return results
        

if __name__ == '__main__':

    try:
        path_to_config = Path("./configs/basic-config.yaml").absolute()
        config = load_yaml(path_to_config)
    except FileNotFoundError:
        raise SystemError("Could not find config file in the directory")

    engine = Engine(config=config)

    engine.load_configuration_and_set_parameters()
    engine.start_rollout_and_learn()
