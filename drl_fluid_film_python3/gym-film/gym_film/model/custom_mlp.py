import param
from stable_baselines.common.policies import FeedForwardPolicy
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

net_arch = param.arch
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=net_arch,
                                                          vf=net_arch)],
                                           feature_extraction="mlp")