# 🚁 RL, DMP and LMPC-Inspired Control for Multi-Quadrotor Systems

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Simulator](https://img.shields.io/badge/simulator-gym--pybullet--drones-orange)](https://github.com/utiasDSL/gym-pybullet-drones)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project demonstrates **Reinforcement Learning (RL) with Dynamic Movement Primitives (DMP)** combined with an **LMPC-inspired outer-loop controller** for **single and multi-quadrotor systems**.

- **RL + DMP** learn smooth Cartesian trajectories with obstacle avoidance  
- **LMPC-inspired velocity control** tracks trajectories without directly commanding positions  
- **gym-pybullet-drones** provides quadrotor dynamics and low-level PID control in Pybullet  
- Supports **multi-agent learning**, **trajectory diversity**, and **Low Level Control under constraints**

---
The Reinforcement Learning formulation is based on:
> Petar Kormushev, Sylvain Calinon and Darwin G. Caldwell  
> ["Robot Motor Skill Coordination with EM-based Reinforcement Learning."](https://www.researchgate.net/publication/224199135_Robot_Motor_Skill_Coordination_with_EM-based_Reinforcement_Learning) (2010)
Is it implemented with numba for fast computation.

The LMPC problem formulation is based on:  
> Alberto, Nicolas Torres, et al.  
> ["Linear Model Predictive Control in SE(3) for online trajectory planning in dynamic workspaces."](https://hal.science/hal-03790059/document) (2022)

The gym-pybullet-drones can be found at https://utiasdsl.github.io/gym-pybullet-drones/ and is based on:
> Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig  
> ["Learning to Fly---a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadcopter Control."](https://arxiv.org/abs/2103.02142) (2021)

---
A more detailed repository for the LMPC formulation can be found at: https://github.com/fleurssauvages/LMPC_for_Manipulators

A more detailed repository for the RL formulation, using a 7 DoF robot arm can be found at: https://github.com/fleurssauvages/RL_DMP

---

## 📂 Project Structure

```
├── RL/                     # RL repo, see the global repo for more details
├── MPC/                    # LMPC repo, see the global repo for more details
├── gym_pybullet_drones/    # forked from the named repo for simulation, needs to be donwloaded separately
├── drone_flock_test.py     # a simple test to check the installation
├── traj_RL.py              # computing and testing trajectories with multi-agents
├── README.md
```

---

## ⚡ Installation
You need to download and install gym-drones-pybullet. See their github for more info.

```bash
pip install pybullet numpy scipy matplotlib numba
pip install "numpy<=2.3"
```

---

## 🎥 Demos

<div align="center">

### 🔹 Reinforcement learning for path finding
<img src="images/RL.gif" width="800" alt="RL">

</div>
---

## TO DO
Add the LMPC to compute desired cartesian velocity with constraints

Add a dynamic avoidance of a moving obstacle

## 🚀 Run the Simulations

- **Simple Quadrotor motion**  
  ```bash
  python drone_flock_test.py
  ```
  Simple test to see if gym-pybullet-drones is correctly installed.

- **RL Traj for Multi-Agent**  
  ```bash
  python traj_RL.py
  ```
  Compute a multiagent trajectory (5 drones) avoiding a static circular obstacle.
---

## 📜 License
This project is licensed under the [MIT License](LICENSE).  

---

## ⭐ Acknowledgments
- Inspired by the work of Alberto, Nicolas Torres, et al. (2022).
- Inspired by the work of Petar Kormushev, Sylvain Calinon and Darwin G. Caldwell (2010)
- Using the simulator and work of Jacopo Panerati et al (2021)