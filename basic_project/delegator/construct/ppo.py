from basic_project.common import methods, constants
from delegator.abstract.base_construct import BaseConstruct
from ray.rllib.algorithms.ppo.ppo import PPOConfig


class PPOConstruct(BaseConstruct):
    def __init__(self, construct_registry_directive: dict):
        self._construct_registry_directive = construct_registry_directive
        self._construct_configuration = None

    @classmethod
    def from_construct_registry_directive(cls, construct_registry_directive: str):
        instance = cls(construct_registry_directive)
        path_to_construct_file = construct_registry_directive.get("path_to_construct_file", None)
        construct_file_path = constants.Directories.TRAINABLE_CONFIG_DIR.value / path_to_construct_file
        instance._construct_configuration = methods.load_yaml(construct_file_path)
        return instance

    def _env_config(self):
        return {
            "env": self._construct_configuration["env-directive"]["prefix"]["choice"],
        }

    def _framework_config(self):
        return {
            "framework": self._construct_configuration["ppo-directive"]["framework"]["choice"]
        }

    def _rollouts_config(self):
        return {
            "num_rollout_workers": self._construct_configuration["ppo-directive"]["rollouts"]["num-workers"]["choice"],
            "num_envs_per_worker": self._construct_configuration["ppo-directive"]["rollouts"]["num-envs-per-worker"]["choice"],
            "rollout_fragment_length": self._construct_configuration["ppo-directive"]["rollouts"]["rollout_fragment_length"]["choice"],
        }

    def _training_config(self):
        return {
            "lr": self._construct_configuration["ppo-directive"]["training"]["lr"]["choice"],
            "lambda_": self._construct_configuration["ppo-directive"]["training"]["lambda_"]["choice"],
            "gamma": self._construct_configuration["ppo-directive"]["training"]["gamma"]["choice"],
            "sgd_minibatch_size": self._construct_configuration["ppo-directive"]["training"]["sgd_minibatch_size"]["choice"],
            "use_gae": self._construct_configuration["ppo-directive"]["training"]["use_gae"]["choice"],
            "train_batch_size": self._construct_configuration["ppo-directive"]["training"]["train_batch_size"]["choice"],
            "num_sgd_iter": self._construct_configuration["ppo-directive"]["training"]["num_sgd_iter"]["choice"],
            "clip_param": self._construct_configuration["ppo-directive"]["training"]["clip_param"]["choice"],
            "model": self._construct_configuration["ppo-directive"]["training"]["model"]["choice"],
        }

    def _resources_config(self):
        return {
            "num_gpus": self._construct_configuration["ppo-directive"]["resources"]["num_gpus"]["choice"],
        }

    def _evaluation_config(self):
        return {
            "evaluation_interval": self._construct_configuration["ppo-directive"]["evaluation"]["evaluation-interval"]["choice"],
            "evaluation_duration": self._construct_configuration["ppo-directive"]["evaluation"]["evaluation-duration"]["choice"],
        }

    def commit(self):
        construct = PPOConfig()

        env = self._env_config()
        construct.environment(**env)

        framework = self._framework_config()
        construct.framework(**framework)

        rollouts = self._rollouts_config()
        construct.rollouts(**rollouts)

        training = self._training_config()
        construct.training(**training)

        resources = self._resources_config()
        construct.resources(**resources)

        evaluation = self._evaluation_config()
        construct.evaluation(**evaluation)

        return construct.to_dict()
