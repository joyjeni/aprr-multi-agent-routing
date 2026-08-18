---
title: APRR CTGR FTDR Live Routing Demo
emoji: 🧭
colorFrom: red
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# APRR / CTGR / FTDR — Live Multi-Agent Routing Demo

Interactive demo for Objective 2 of a PhD research project on adaptive
multi-agent LLM routing.

This Space clones and runs the **real router source code** from
[aprr-multi-agent-routing](https://github.com/joyjeni/aprr-multi-agent-routing),
warms the routers up online on the real
[ToolBench-v1 dataset](https://huggingface.co/datasets/tuandunghcmut/toolbench-v1)
(base paper: [Qin et al., ToolLLM, ICLR 2024](https://arxiv.org/abs/2307.16789)),
and lets you type a query to see **APRR-core**, **CTGR** (Chain-of-Thought Gated
Router), and **FTDR** (Functional-Token Dispatch Router) make a live routing
decision — with an optional simulated Objective 1 fusion signal.

- Objective 2 repo (routing code): https://github.com/joyjeni/aprr-multi-agent-routing
- Objective 1 repo (session-aware reranking): https://github.com/joyjeni/session-aware-toolbench-rerank
- Results dashboard: https://aprr-multi-agent-routing.vercel.app
