# TODO add/fix docstrings
# TODO add comments

import math
from typing import Dict, Tuple

import torch

from stable_baselines3.common.utils import get_device

from cadam import CAdam


def prepare_cbp_kwargs(net, net_args):
    return {
        "linear_layers": [
            list(net.dueling.value_branch[::2].children()),
            list(net.dueling.advantage_branch[::2].children()),
        ],
        "activation_layers": [
            list(net.dueling.value_branch[1::2].children()),
            list(net.dueling.advantage_branch[1::2].children()),
        ],
        #        'is_training': net.is_training
    }


@torch.jit.script
def _hook_calcs(cbp_vals: Dict[str, torch.Tensor], out: torch.Tensor, eta: float):
    # NOTE Seems CBP is only described for sequential input with gradient updates at each step.
    #      Since PPO is based on batched environment data, changes have to be made
    #      I will therefore work with means over the baches
    # cbp_vals['age'].add_(1)
    cbp_vals["h"].copy_(out.mean(0))
    cbp_vals["fhat"].copy_(cbp_vals["f"] / (1 - eta ** cbp_vals["age"]))
    cbp_vals["f"].mul_(eta).add_((1 - eta) * cbp_vals["h"])


@torch.jit.ignore
def linear_sample_weights(size: Tuple[int, int], device: torch.device):
    sample = torch.empty(size, device=device)
    torch.nn.init.kaiming_uniform_(sample, a=math.sqrt(5))
    return sample


def _generate_step_function(sample_weights):
    """
    When resetting parameters, the new values should be sampled from the original distribution.
    Unfortunately, torch JIT does not support functions as arguments, preventing a generic implementation.
    As a work-around, we can instantiate a new JIT function with the desired sampling function "hard coded" at instantiation.
    """

    @torch.jit.script
    def step_calcs(
        cbp_vals: Dict[str, torch.Tensor],
        pre_state: Dict[str, torch.Tensor],
        post_state: Dict[str, torch.Tensor],
        pre_linear: torch.nn.Parameter,
        post_linear: torch.nn.Parameter,
        eta: float,
        m: int,
        rho: float,
        eps: float,
    ):
        pre_w = pre_linear.detach().abs().sum(1).add(eps)  # avoid division by zero
        post_w = post_linear.detach().abs().sum(0)

        y = (cbp_vals["h"] - cbp_vals["fhat"]).abs().mul(post_w).div(pre_w)
        cbp_vals["u"].mul_(eta).add_((1 - eta) * y)

        uhat = cbp_vals["u"] / (1 - eta ** cbp_vals["age"])

        eligible = cbp_vals["age"] > m
        if (
            eligible.any() and torch.rand(1) < len(uhat) * rho
        ):  # use n_l* rho as a probability of replacing a single feature
            ascending = uhat.argsort()
            r = ascending[
                eligible[ascending]
            ]  # sort eligible indices according to their utility
            # r = r[:math.ceil(uhat.shape[0]*self.rho)]  # choose top k worst performing features    # using ceil because otherwise nothing ever gets reset int(256*10**-4)=0
            r = r[[0]]  # choose the worst feature

            pre_linear.index_copy_(
                0, r, sample_weights((len(r), pre_linear.shape[1]), pre_linear.device)
            )
            post_linear.index_fill_(1, r, 0.0)

            cbp_vals["u"].index_fill_(0, r, 0.0)
            cbp_vals["f"].index_fill_(0, r, 0.0)
            cbp_vals["age"].index_fill_(0, r, 0)

            ### Adam resets
            pre_state["step"].index_fill_(0, r, 0)
            pre_state["exp_avg"].index_fill_(0, r, 0.0)
            pre_state["exp_avg_sq"].index_fill_(0, r, 0.0)

            post_state["step"].index_fill_(1, r, 0)
            post_state["exp_avg"].index_fill_(1, r, 0.0)
            post_state["exp_avg_sq"].index_fill_(1, r, 0.0)

        # moved her from _hook_calcs
        # rainbow implementation doesn't differentiate between train and eval, so we have no way to know in which state we are.
        # by doing it here we know every addition corresponds with an update
        cbp_vals["age"].add_(1)

    return step_calcs


class CBP(CAdam):
    """
    Open questions:
        How should batches be dealth with?
            For now I calculate the mean over the batch and handle that like in the sequential case
        How many features are actually replaced every iteration? Their n_l and rho don't seem to work, as 256 * 10**-4 < 1. Is this supposed to be a probability?
            # For now math.ceil is used, so every iteration 1 usit is replaced. This doesn't make sense, since when n_l < m then the features are just replaced in order as they mature.
            Changed to using n_l * rho as a probability of replacing the worst performing feature
    """

    def __init__(
        self,
        params,  # all parameters to be optimized by Adam
        linear_layers,  # List[List[Linear]], a list of linearities for each separate network (policy, value, ...), in the order they are executed
        activation_layers,  # List[List[Activation]], a list of activation layers for each separate network (policy, value, ...), in the order they are executed. Forward hooks are added to these
        eta=0.99,  # running average discount factor
        m=int(
            5e3
        ),  # maturity threshold, only features with age > m are elligible to be replaced
        rho=10**-4,  # replacement rate, controls how frequently features are replaced
        sample_weights=None,  # functiion, take size and device as input and return a tensor of the given size with newly initialized weights
        eps=1e-8,  # small additive value to avoid division by zero
        device="auto",
        #                 is_training=None,       # function returning bool, controls when cbp calculations are done in the forward pass
        **kwargs,
    ):
        super(CBP, self).__init__(params, eps=eps, **kwargs)
        self.linear_layers = linear_layers
        self.activation_layers = activation_layers
        self.cbp_vals = {}
        self.eta = eta
        self.m = m
        self.rho = rho

        self.dev = get_device(device)
        #        self.is_training = is_training

        assert len(self.linear_layers) == len(self.activation_layers)
        for linears, activations in zip(self.linear_layers, self.activation_layers):
            self._add_hooks(linears, activations)

        # if sample_weights is None:
        #    def sample_weights(size, device):
        #        sample = torch.empty(size, device=device)
        #        torch.nn.init.kaiming_uniform_(sample, a=math.sqrt(5))
        #        return sample
        # self.sample_weights = sample_weights
        self.eps = eps

        self.step_calcs = _generate_step_function(linear_sample_weights)

    @torch.no_grad()
    def step(self):
        super(CBP, self).step()
        for linears in self.linear_layers:  # cycle through models
            for current_linear, next_linear in zip(
                linears[:-1], linears[1:]
            ):  # cycle through layers
                cbp_vals = self.cbp_vals[current_linear]
                pre_state = self.state[current_linear.weight]
                post_state = self.state[next_linear.weight]

                self.step_calcs(
                    cbp_vals,
                    pre_state,
                    post_state,
                    current_linear.weight,
                    next_linear.weight,
                    self.eta,
                    self.m,
                    self.rho,
                    self.eps,
                )  # self.sample_weights)

    def _hook_gen(self, linear_layer):
        num_units = linear_layer.weight.shape[0]
        self.cbp_vals[linear_layer] = {
            "age": torch.ones(
                num_units, dtype=int, device=self.dev, requires_grad=False
            ),
            "h": torch.zeros(num_units, device=self.dev, requires_grad=False),
            "f": torch.zeros(num_units, device=self.dev, requires_grad=False),
            "fhat": torch.zeros(num_units, device=self.dev, requires_grad=False),
            "u": torch.zeros(num_units, device=self.dev, requires_grad=False),
        }

        def hook(mod, inp, out):
            # if (self.is_training is not None and self.is_training()) or (self.is_training is None and mod.training):
            if mod.training:
                cbp_vals = self.cbp_vals[linear_layer]
                with torch.no_grad():
                    _hook_calcs(cbp_vals, out, self.eta)

        return hook

    def _add_hooks(self, linears, activations):
        assert (
            len(linears) == len(activations) + 1
        )  # 1 to 1 linear and activation, plus output linear
        for lin, act in zip(linears, activations):
            act.register_forward_hook(self._hook_gen(lin))
