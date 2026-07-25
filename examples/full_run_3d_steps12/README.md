# Full-scale 3D DP run (steps=12, 193,551 nodes)

A real, uncapped `--algo dp` run over a custom map built with the map editor GUI,
kept here (rather than in the gitignored `outputs/` folder) as a concrete,
reproducible data point referenced in `docs/DP_THERAPY_PLANNER.md` §10c.

**Command that produced it:**

```bash
python main_demo_3d.py --algo dp --steps 12 --step-size 1.0 --seed 0 \
    --start-x 6.0 --start-y 0.0 --start-z -1.0 --start-orientation 13 \
    --eps-len 1.0 --eps-cov 1.0 --eps-dam 1.0 --eps-ndose 1.0
```

(The map itself is whatever `sim3d/demo_map3d.py` contained at the time — see
`run.log` for the full log of that run.)

**Headline numbers** (from `dp_stats.json`): 193,551 nodes, 666,035 edges,
75,315,424 dominance comparisons, peak frontier 144, final frontier 36 plans,
peak memory 621.84MB, 897.1s total pipeline time (68% of it in `dp.compute`).

**Files:**
- `dp_stats.json` — per-phase timing, aggregate comparison/frontier/memory stats.
- `dp_node_stats.csv` — the same breakdown per individual graph node (193,551 rows).
- `frontier_3d.pkl` — the saved Pareto frontier; reopen the interactive dashboard
  without recomputing anything:
  ```bash
  python replay_dashboard.py examples/full_run_3d_steps12/frontier_3d.pkl
  ```
- `dashboard_3d.gif` / `simulation_3d.gif` — pre-rendered animations from the run.
- `path.json` — the chosen (Min Damage) plan's replayable path.
- `run.log` — the full run log.
