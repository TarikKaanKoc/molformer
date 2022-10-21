#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement the causally masked linear attention as a recurrent model."""

import torch
from torch.nn import Module

from ....attention_registry import RecurrentAttentionRegistry, Optional, Int, \
    Callable, EventDispatcherInstance
from ....events import EventDispatcher
from ....feature_maps import elu_feature_map
from ..._utils import check_state


class RecurrentLinearAttentionPPLM(Module):
    """Implement fast_transformers.attention.causal_linear_attention as a
    fixed-dimensional state recurrent model.

    See fast_transformers.attention.linear_attention and
    fast_transformers.attention.causal_linear_attention for the general concept
    of replacing the softmax with feature maps.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(RecurrentLinearAttentionPPLM, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, query, key, value, state=None, memory=None):
        # Normalize state/memory
        state = check_state(state, memory)
        
        # If this is a new sequence reinitialize the feature map
        if state is None:
            self.feature_map.new_feature_map(query.device)

        # Apply the feature map to the query and key
        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)

        # Extract some shapes
        N, H, D = Q.shape
        _, _, M = value.shape

        # Extract the memory or initialize it
        if state is None:
            K = K.unsqueeze(-1)
            value = value.unsqueeze(-1)
            #print(K.size())
            #print(value.size())
        else:
            #state is a tuple that contains the concatination of all time steps of key and value
            # where index 0 is all keys and index 1 is all values
            K = torch.cat([state[0], K.unsqueeze(-1)], dim=-1)
            value = torch.cat([state[1], value.unsqueeze(-1)], dim=-1)


        # Update the internal state
        #sum all past K
        print('K')
        print(K.size())
        print('value')
        print(value.size())
        Zi = torch.einsum("nhdj->nhd", K)
        #matrix product for all past steps of K and value
        Si = torch.einsum("nhdj,nhmj->nhdm", K, value)
        print('Zi')
        print(Zi.size())
        print('Si')
        print(Si.size())

        # Ensure the batch size did not change
        if len(Si) != N:
            raise ValueError("The batch size changed during iteration")

        # Compute the output
        Z = 1. / (torch.einsum("nhd,nhd->nh", Q, Zi) + self.eps)
        V = torch.einsum("nhd,nhdm,nh->nhm", Q, Si, Z)

        return V, (K, value)


# Register the attention implementation so that it becomes available in our
# builders
RecurrentAttentionRegistry.register(
    "linear-pplm", RecurrentLinearAttentionPPLM,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
RecurrentAttentionRegistry.register(
    "causal-linear-pplm", RecurrentLinearAttentionPPLM,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
