import math
from typing import Dict, Tuple, Callable, List

import torch
from torch.optim.optimizer import ParamsT

from stable_baselines3.common.utils import get_device

from cadam import CAdam


@torch.jit.script
def _hook_calcs(cbp_vals: Dict[str, torch.Tensor], out: torch.Tensor, eta: float):
    """Calculate and store the average activation output and running average of that.

    **Note**: Seems CBP is only described for sequential input with gradient updates at each step.
              As most RL is based on batched environment data, changes have to be made
              We will therefore work with means over the baches

    Args:
        cbp_vals (Dict[str, torch.Tensor]): values being tracked for a specififc linear layer
        out (torch.Tensor): output of the activation function following the linear layer
        eta (float): running average decay rate
    """
    # average outputs
    cbp_vals["h"].copy_(out.mean(0))

    # running average of outputs
    cbp_vals["f"].mul_(eta).add_((1 - eta) * cbp_vals["h"])

    # age is updated in step_calcs, to differentiate between training and evaluation


@torch.jit.ignore
def linear_sample_weights(size: Tuple[int, int], device: torch.device):
    sample = torch.empty(size, device=device)
    torch.nn.init.kaiming_uniform_(sample, a=math.sqrt(5))
    return sample


def _generate_step_function(sample_weights) -> torch.jit.ScriptFunction:
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
        # calculate the magnitudes of incoming and outgoing weights
        pre_w = pre_linear.detach().abs().sum(1)
        post_w = post_linear.detach().abs().sum(0)

        # bias corrected running average of outputs
        fhat = cbp_vals["f"] / (1 - eta ** cbp_vals["age"])

        # current total utility
        y = (cbp_vals["h"] - fhat).abs().mul(post_w).div(pre_w.add(eps))

        # running average total utility
        cbp_vals["u"].mul_(eta).add_((1 - eta) * y)

        # bias corrected running average total utility
        uhat = cbp_vals["u"] / (1 - eta ** cbp_vals["age"])

        # find all units that are older than the minimum age
        eligible = cbp_vals["age"] > m

        # use n_l* rho as a probability of replacing a single feature
        if eligible.any() and torch.rand(1) < len(uhat) * rho:
            # sort eligible indices according to their utility
            ascending = uhat.argsort()
            r = ascending[eligible[ascending]]

            r = r[[0]]  # choose the worst feature

            # sample new weights for ingoing connections
            pre_linear.index_copy_(
                0, r, sample_weights((len(r), pre_linear.shape[1]), pre_linear.device)
            )

            # disable outgoing connections by setting their weights to zero
            post_linear.index_fill_(1, r, 0.0)

            # reset tracked values
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

        cbp_vals["age"].add_(1)

    return step_calcs


class CBP(CAdam):
    """
    Continual Backprop extension for the Adam Optimizer

    Open questions:
        - How should batches be dealt with?
            - For now we calculate the mean over the batch and handle that like in the sequential case
        - How many features are actually replaced every iteration? Their n_l and rho don't seem to work, as 256 * 10**-4 < 1. Is this supposed to be a probability?
            - Changed to using n_l * rho as a probability of replacing the worst performing feature
    """

    def __init__(
        self,
        params: ParamsT,
        linear_layers: List[List[torch.nn.Linear]],
        activation_layers: List[List[torch.nn.Module]],
        eta: float = 0.99,
        m: int = int(5e3),
        rho: float = 10**-4,
        sample_weights: Callable = linear_sample_weights,
        eps: float = 1e-8,
        device: torch.device | str = "auto",
        **kwargs,
    ):
        """
        Args:
            params (ParamsT): Parameters to be optimized
            linear_layers (List[List[torch.nn.Linear]]): a list of linearities
                for each separate network (policy, value, ...), in the order they are executed
            activation_layers (List[List[torch.nn.Module]]): a list of activation layers
                for each separate network (policy, value, ...), in the order they are executed.
                Forward hooks are added to these
            eta (float, optional): running average discount factor. Defaults to 0.99.
            m (int, optional): maturity threshold, only features with ``age > m`` are elligible to be replaced. Defaults to int(5e3).
            rho (float, optional): replacement rate, controls how frequently features are replaced. Defaults to 10**-4.
            sample_weights (Callable, optional): functiion that takes a size and a device as input
                and returna a tensor of the given size with newly initialized weights.
                Defaults to linear_sample_weights.
            eps (float, optional): small additive value to avoid division by zero. Defaults to 1e-8.
            device (torch.device | str, optional): Pytorch device on which to store tensors. Defaults to "auto".
        """
        super(CBP, self).__init__(params, eps=eps, **kwargs)

        self.linear_layers = linear_layers
        self.activation_layers = activation_layers

        self.cbp_vals = {}

        self.eta = eta
        self.m = m
        self.rho = rho

        self.dev = get_device(device)

        # add forward hooks to all activation functions, which then store values relevant to cbp
        # in the cbp_vals dictionary, with the linear layer as the key
        assert len(self.linear_layers) == len(self.activation_layers)
        for linears, activations in zip(self.linear_layers, self.activation_layers):
            # 1 to 1 linear and activation
            assert len(linears) == len(activations)

            for lin, act in zip(linears, activations):
                act.register_forward_hook(self._hook_gen(lin))

        self.eps = eps

        self.step_calcs = _generate_step_function(sample_weights)

    @torch.no_grad()
    def step(self):
        # execute normal optimizer step
        super(CBP, self).step()

        # apply Continual Backprop:
        # cycle through models
        for linears in self.linear_layers:
            # cycle through layers
            for current_linear, next_linear in zip(linears[:-1], linears[1:]):
                # prepare variables
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
                )

    def _hook_gen(self, linear_layer: torch.nn.Linear) -> Callable:
        """Set up a forward hook for the given linear layer.

        Also adds a corresponding entry to ``cbp_vals``.

        - ``age`` counts the number of updates since the last reset
        - ``h`` stores the last mean activation output
        - ``f`` and ``u`` are running averages of the activation output and total utility respectively

        All of these are vectors with an entry per unit in the linear layer"""
        n_units = linear_layer.weight.shape[0]
        self.cbp_vals[linear_layer] = {
            "age": torch.ones(n_units, dtype=int, device=self.dev, requires_grad=False),
            "h": torch.zeros(n_units, device=self.dev, requires_grad=False),
            "f": torch.zeros(n_units, device=self.dev, requires_grad=False),
            "u": torch.zeros(n_units, device=self.dev, requires_grad=False),
        }

        # the cbp calculations are only interested in the activation output but are
        # used to reset the preceding linear layer.
        # therefore, we add this hook to the activation layer, then
        # store the calculation results in the ``cbp_vals`` entry for the linear layer
        def hook(mod, inp, activation_output):
            if mod.training:
                cbp_vals = self.cbp_vals[linear_layer]
                with torch.no_grad():
                    _hook_calcs(cbp_vals, activation_output, self.eta)

        return hook
