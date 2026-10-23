"""SB3 policies used by the lab beyond the default MlpPolicy.

CriticExtrasPolicy is an asymmetric actor-critic: the observation ends with simulator facts the robot cannot
measure (G1 real-robot profile: base linear velocity and pelvis height, `critic_extras`), and only the value
network sees them. The actor's features are the same vector with those entries zeroed, so the deployed
policy depends on the robot's own measurements alone. Saved stages stay ordinary SB3 archives; loading needs
this module importable.
"""
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor


class ActorInputs(BaseFeaturesExtractor):
    """Keeps the first `actor_inputs` observations and zeroes the rest (same width, no parameters)."""

    def __init__(self, observation_space, actor_inputs):
        super().__init__(observation_space, observation_space.shape[0])
        self.actor_inputs = int(actor_inputs)

    def forward(self, observations):
        visible = observations[:, :self.actor_inputs]
        return torch.cat((visible, torch.zeros_like(observations[:, self.actor_inputs:])), dim=1)


class CriticExtrasPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, actor_inputs=None, **kwargs):
        if actor_inputs is None or not 0 < actor_inputs <= observation_space.shape[0]:
            raise ValueError('actor_inputs must count the leading observations the actor may see')
        kwargs.update(share_features_extractor=False, features_extractor_class=ActorInputs,
                      features_extractor_kwargs=dict(actor_inputs=actor_inputs))
        self.actor_inputs = int(actor_inputs)
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # Both extractors are parameter-free, so swapping the critic's after construction leaves the
        # optimizer and the saved state dict unchanged.
        self.vf_features_extractor = FlattenExtractor(observation_space)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(actor_inputs=self.actor_inputs)
        for key in ('share_features_extractor', 'features_extractor_class', 'features_extractor_kwargs'):
            data.pop(key, None)  # set by __init__ itself
        return data
