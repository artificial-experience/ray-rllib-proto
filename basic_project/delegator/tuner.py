from ray import air, tune
from common import methods
from .trainable import TrainableConstructDelegator

class TunerDelegator:
     
    def __init__(self,
        construct_directive: dict,
        tuner_directive: dict
        ):
        self.construct_directive = construct_directive
        self.tuner_directive = tuner_directive

        # return _param_space and ray_trainable_prefix
        self._trainable_construct_delegator = None

        self._ray_trainable_prefix = None
        self._param_space = None
        self._run_config = None

        #TODO: create this part of a workflow
        self._tune_config = None

    @classmethod
    def from_trial_directive(cls, construct_directive: dict, tuner_directive: dict):
        instance = cls(construct_directive, tuner_directive)
        instance._trainable_construct_delegator = TrainableConstructDelegator.from_construct_directive(
            construct_directive=construct_directive
        )
        return instance

    def _setup_run_config(self):
        pass

    def _setup_trainable_prefix_and_param_space(self):
        self._param_space = self._trainable_construct_delegator.delegate()
        self._ray_trainable_prefix = self._trainable_construct_delegator.target_trainable_ray_prefix

    def delegate_tuner_entity(self):
        """Instantiate and return tuner entity ready for training"""
        self._setup_run_config()
        self._setup_trainable_prefix_and_param_space()
        tuner = tune.Tuner(
            trainable=self._ray_trainable_prefix,
            param_space=self._param_space,
            run_config=self._run_config,
        )
        return tuner 
