"""Inference-only loader for the released GCBF+ actor checkpoint.

Training, GCBF v0, benchmark solvers and online QP refinement are intentionally
not part of this simulator. The policy equation matches GCBFPlus.act():
    u = 2 * actor.get_action(params, graph) + env.u_ref(graph)
"""
from __future__ import annotations

import os
import pickle

import jax.numpy as jnp
import jax.random as jr

from gcbfplus.algo.module.policy import DeterministicPolicy
from gcbfplus.utils.graph import GraphsTuple


class GCBFPlusPolicy:
    def __init__(self, env, node_dim: int, edge_dim: int, state_dim: int, action_dim: int,
                 n_agents: int, gnn_layers: int = 1, seed: int = 0, **_unused):
        self._env = env
        self.n_agents = int(n_agents)
        self.actor = DeterministicPolicy(
            node_dim=node_dim, edge_dim=edge_dim, action_dim=action_dim,
            n_agents=n_agents, gnn_layers=gnn_layers)
        graph = GraphsTuple(
            nodes=jnp.zeros((n_agents, node_dim)),
            edges=jnp.zeros((n_agents, edge_dim)),
            states=jnp.zeros((n_agents, state_dim)),
            n_node=jnp.array(n_agents),
            n_edge=jnp.array(n_agents),
            senders=jnp.arange(n_agents),
            receivers=jnp.arange(n_agents),
            node_type=jnp.zeros((n_agents,)),
            env_states=jnp.zeros((n_agents,)),
        )
        self.actor_params = self.actor.net.init(jr.PRNGKey(seed), graph, self.n_agents)

    def act(self, graph):
        return 2 * self.actor.get_action(self.actor_params, graph) + self._env.u_ref(graph)

    def load(self, load_dir: str, step: int):
        actor_path = os.path.join(load_dir, str(step), "actor.pkl")
        with open(actor_path, "rb") as f:
            self.actor_params = pickle.load(f)
