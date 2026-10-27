# GCBF+ LinearDrone Simulator

A lightweight interactive simulator for the **GCBF+** learned multi-agent controller, focused on the LinearDrone environment described in: S. Zhang, O. So, K. Garg, and C. Fan, “GCBF+: A Neural Graph Control Barrier Function Framework for Distributed Safe Multiagent Control,” *IEEE Transactions on Robotics*, vol. 41, 2025, pp. 1533–1552. DOI: [10.1109/TRO.2025.3530348](https://doi.org/10.1109/TRO.2025.3530348). The original project site is [MIT-REALM/gcbfplus](https://mit-realm.github.io/gcbfplus/).

## Run

```bash
python -m pip install -r requirements.txt
python run_ui.py
```

The UI contains only the GCBF+ simulation, with a mode selector for **2D LinearDrone** and **3D LinearDrone**. Use **Run**, **Pause**, **Step**, and **Reset** in the top toolbar.

## Simulation modes

- **3D LinearDrone:** uses the released pretrained GCBF+ LinearDrone checkpoint. Rotate the 3D matplotlib view by dragging and zoom with the scroll wheel.
- **2D LinearDrone:** embeds the same 3D LinearDrone model in a plane: each agent and obstacle stays at a fixed height and vertical control is set to zero. It is a practical planar view of the released 3D policy, not a separately trained 2D checkpoint.

The original paper evaluates LinearDrone as a 3D environment; the paper's 2D environments are other dynamics classes. The planar mode here is a UI/simulation adaptation so both requested views can run with the supplied pretrained LinearDrone model.

## UI controls

- **Scenario:** configure arena size, agent/obstacle counts, random placement, obstacle/target velocities, random seed, and load/save scenario JSON.
- **Agents and obstacles:** edit starts, targets, obstacle centers/radii, target velocities, and obstacle velocities in the tables.
- **Parameters:** adjust drone radius, communication/LiDAR range, LiDAR beams, time step, and optional graph-size caps.
- **2D canvas tools:** select/move, add agent (start then target), add obstacle (click and drag for radius), delete, and edit obstacle/target velocity arrows.

Moving obstacles and targets travel with constant velocity and bounce at arena boundaries. The learned policy/checkpoint weights are fixed; training and benchmarking tools have intentionally been removed.

## Checkpoint

The app expects the released LinearDrone checkpoint at:

```text
pretrained/LinearDrone/gcbf+/config.yaml
pretrained/LinearDrone/gcbf+/models/1000/actor.pkl
```

Only the actor weights are needed for live policy inference; the unused training-time `cbf.pkl` weights have been removed to keep the package lean. The policy action equation is the GCBF+ inference equation `u = 2 * actor(graph) + u_ref(graph)`. The simulator retains the learned policy and its environment/graph implementation, but omits training, GCBF v0, benchmark baselines, Crazyflie dynamics, and unrelated pretrained weights.

## Attribution and license

This project is derived from the MIT-REALM GCBF+ implementation and contains the corresponding pretrained checkpoint files. The upstream MIT license and its copyright notice are retained in `LICENSE`, as required by the license terms. Please consult the paper and upstream project for the scientific method and original implementation details.
