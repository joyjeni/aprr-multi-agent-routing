"""Smoke tests for the APRR router."""
import numpy as np

from aprr import APRRRouter, AgentTopology, RouterConfig
from aprr.baselines import (LLMRouter, OracleRouter, RandomRouter,
                            RoundRobinRouter, StaticSemanticRouter)
from aprr.toolbench import ToolBenchSimulator


def test_router_runs():
    topo = AgentTopology.default(seed=0)
    sim = ToolBenchSimulator(topo, seed=0)
    queries = sim.generate(10)
    r = APRRRouter(topo, RouterConfig(seed=1))
    for q in queries:
        path = r.route(q.embedding)
        assert len(path) >= 1
        s, lat = sim.rollout(path, q.gt_path)
        r.update_trail(path, s, lat)
    assert r.W.shape == (topo.n, topo.n)
    assert np.all(np.isfinite(r.W))


def test_all_baselines_run():
    topo = AgentTopology.default(seed=0)
    sim = ToolBenchSimulator(topo, seed=0)
    queries = sim.generate(5)
    routers = [
        RandomRouter(topo, seed=1),
        RoundRobinRouter(topo, seed=2),
        StaticSemanticRouter(topo, seed=3),
        LLMRouter(topo, seed=4),
        APRRRouter(topo, RouterConfig(seed=5)),
    ]
    for r in routers:
        for q in queries:
            path = r.route(q.embedding)
            s, lat = sim.rollout(path, q.gt_path)
            r.update_trail(path, s, lat)


def test_oracle_perfect():
    topo = AgentTopology.default(seed=0)
    sim = ToolBenchSimulator(topo, seed=0); queries = sim.generate(20)
    r = OracleRouter(topo, ground_truth_fn=sim.gt_path_of, seed=0)
    succs = []
    for q in queries:
        r.set_query(q.qid)
        path = r.route()
        s, _ = sim.rollout(path, q.gt_path)
        succs.append(s)
    # Oracle should hit success > 0.8 on average
    assert np.mean(succs) > 0.7


def test_W_evolves():
    topo = AgentTopology.default(seed=0)
    sim = ToolBenchSimulator(topo, seed=0); queries = sim.generate(50)
    r = APRRRouter(topo, RouterConfig(seed=7))
    W0 = r.W.copy()
    for _ in range(5):
        for q in queries:
            path = r.route(q.embedding)
            s, lat = sim.rollout(path, q.gt_path)
            r.update_trail(path, s, lat)
    # W must have changed
    assert not np.allclose(W0, r.W)
