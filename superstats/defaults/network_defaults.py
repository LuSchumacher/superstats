import bayesflow as bf

from superstats.networks import RecurrentNet

DEFAULT_SUMMARY_NETWORK = RecurrentNet(
    recurrent_type="gru",
    hidden_dim=128
)

DEFAULT_INFERENCE_NETWORK = bf.networks.StableConsistencyModel(
    subnet_kwargs={"widths": (128,)*3}
)