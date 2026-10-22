# Graph Neural Network simulator: three colliding balls

A minimal, fully inspectable version of the **Graph Network-based Simulator (GNS)** from
Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks* (ICML 2020,
[arXiv:2002.09405](https://arxiv.org/abs/2002.09405)), in pure JAX like the rest of this repository.

The physical system is three balls with a spring-damper contact force. It is small on purpose: with a
latent size of 8 and 2 message-passing steps, every node matrix, edge matrix and message fits on screen.

## Start here

[`notebooks/gnn_explained.ipynb`](notebooks/gnn_explained.ipynb) walks through:

1. the ground-truth simulator,
2. how one moment becomes a graph (node features, edge features, mask, adjacency),
3. one forward pass, matrix by matrix (encoder → edge function → sum → node function → decoder),
4. training with GNS-style noise injection, keeping a weight snapshot every 100 steps,
5. an **interactive explorer** with sliders for *training step*, *simulation time* and
   *message-passing step*, showing every matrix plus true vs predicted accelerations,
6. the force law the edge function rediscovers,
7. the trained GNN used as a simulator, including on 5 balls (trained on 3).

The sliders need a running Jupyter kernel. The saved outputs (including a static version of the explorer) render on
GitHub.

## Layout

```
gnn/
├── src/
│   ├── physics.py     ground-truth balls: contact force, semi-implicit Euler, random datasets
│   ├── graph.py       particles → graph (radius edges with a mask, relative edge features)
│   ├── model.py       encode-process-decode GNN; apply_with_trace returns every intermediate matrix
│   ├── training.py    normalisation, noise injection, Adam loop with snapshots, learned rollout
│   └── visualize.py   ball/graph drawing and annotated matrix heatmaps
├── notebooks/
│   └── gnn_explained.ipynb
└── requirements.txt
```

## Run

```bash
cd 05-graph-neural-networks/gnn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/gnn_explained.ipynb
```

Training takes about 30 seconds on a laptop CPU.

## Differences from the paper (all for readability)

| | GNS paper | here |
|---|---|---|
| latent size / MLP width | 128 / 128 | 8 / 32 |
| message-passing steps | 10 | 2 |
| node inputs | last 5 velocities, material type, boundary distances | current velocity |
| LayerNorm | yes | no |
| edges | radius graph via neighbour search | all pairs with a radius mask (N is tiny) |
