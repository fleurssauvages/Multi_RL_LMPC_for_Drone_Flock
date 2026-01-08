import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from RL.dmp import MixturePD
from RL.env_reaching import ReachingEnv
from RL.demo_utils import init_from_demo, make_exploration_std, set_axes_equal, make_demo_6D
from RL.multiagent_power_rl import MultiAgentPowerRL
from RL.resample import resample_min_jerk

'''' 
Example of using Multi-Agent Power RL to learn reaching trajectories for multiple drones in a PyBullet environment.
The current script initializes a circular formation of drones, sets up a reaching task with obstacles, and runs the learning algorithm.
The Drones then execute their best learned trajectories in the simulation, with a PID controller.
'''

def main():
    # --- Init Drones and Drone Env ---
    NUM_DRONES = 5
    r = 0.2      # circle radius (meters)
    center = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]) # circle center
    goal = np.array([0.0, 2.0, 0.5, 0.0, 0.0, 0.0])   # reaching goal

    angles = np.linspace(0, 2*np.pi, NUM_DRONES, endpoint=False)

    init_xyzs = np.zeros((NUM_DRONES, 3))
    init_xyzs[:, 0] = center[0] + r * np.cos(angles)   # x
    init_xyzs[:, 1] = center[1] + r * np.sin(angles)   # y
    init_xyzs[:, 2] = center[2]                        # z

    ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(NUM_DRONES)]
    init_rpys = np.zeros((NUM_DRONES, 3)) # initial roll/pitch/yaw

    # Simulation/control rates (common choice: sim 240 Hz, control 48 Hz -> 5 sim steps per control step)
    CTRL_HZ = 200
    dt_ctrl = 1.0 / CTRL_HZ

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=Physics.PYB,
        gui=True,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    sim_duration = 5.0 # seconds
    action = np.zeros((NUM_DRONES, 4))

    # --- Create obstacles in the PyBullet env ---
    centerObs = (center[:3] + goal[:3]) / 2
    radiusObs = np.linalg.norm(goal[:3] - center[:3])/6
    obstacles = [
        {'center': centerObs, 'radius': radiusObs},
    ]

    def create_sphere_obstacle(center, radius):
        collision = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
        )
        visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[1, 0, 0, 0.6],
        )
        body = p.createMultiBody(
            baseMass=0.0,  # static obstacle
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=center.tolist(),
        )
        return body
    
    obstacle_ids = []
    for obs in obstacles:
        obstacle_ids.append(create_sphere_obstacle(obs["center"], obs["radius"]))


    # --- Init RL ---
    np.random.seed(1)

    # --- Parameters ---
    D, K = 6, 4
    duration, dt = 1.0, 0.02
    weight_demo, weight_goal = 0.05, 0.95
    weight_jerk, weight_end_vel = 0.005, 0.05
    n_iterations, rollouts_per_agent = 120, 8
    n_agents = 5
    exploration_std = make_exploration_std(D, K, sigma_pos=0.05, sigma_ori=0.1, sigma_kp=0.02)
    decay = 0.98
    scaling = 0.5

    # --- DMP and environment setup ---
    dmp = MixturePD(D=D, K=K, duration=duration, kp_diag=160.0, vel_gain=12.0)
    demo = make_demo_6D(duration=duration, timesteps=int(duration / dt), curvature=0.05, start=center, goal=goal, normalize=scaling)
    scaling = np.linalg.norm(goal[:3] - center[:3]) / scaling
    dmp = init_from_demo(dmp, demo, kp_diag=80.0)
    goal = demo["x"][-1]

    obstacles = [
        {'center': (centerObs - center[:3]) / scaling, 'radius': radiusObs / scaling},
    ]
    envRL = ReachingEnv(dmp, dt=dt, obstacles=obstacles, demo_traj=demo, goal=goal)

    # --- Multi-agent RL system ---
    population = MultiAgentPowerRL(
        dmp.get_flat_params(),
        exploration_std=exploration_std,
        n_agents=n_agents,
        reuse_top_n=3,
        diversity_strength=0.1 * np.mean(exploration_std)
    )
    # --- Main loop ---
    for it in range(n_iterations):
        population.reset_histories()
        def rollout_job(agent_id):
            agent_local = population.agents[agent_id]
            params_k = agent_local.sample_policy()
            traj, Rk = envRL.simulate_and_return_traj_numba(
                params_k,
                w_demo=weight_demo,
                w_goal=weight_goal,
                w_jerk=weight_jerk,
                w_end_vel=weight_end_vel,
            )
            return agent_id, params_k, traj, Rk

        # Create one job per (agent, rollout)
        jobs = [agent_id for agent_id in range(n_agents) for _ in range(rollouts_per_agent)]

        # Run all rollouts
        results_all = [rollout_job(agent_id) for agent_id in jobs]
        
        # # Assign results per agent
        for agent_id, params_k, _, Rk in results_all:
            agent = population.agents[agent_id]
            agent.add_rollout(params_k, Rk)

        # Update each agent and apply diversity
        population.update_agents()
        population.apply_diversity_pressure(exploration_std=exploration_std*0.1)
        population.update_exploration(exploration_std * (decay ** it))
        population.update_diversity_strength(population.diversity_strength * decay)

    # --- Compute best traj/return per agent ---
    best_trajs_per_agent = []
    best_returns_per_agent = []

    for _, agent in enumerate(population.agents):
        if len(agent.history_returns) == 0:
            best_trajs_per_agent.append(None)
            best_returns_per_agent.append(-np.inf)
            continue

        history_R = np.array(agent.history_returns)
        best_idx_local = np.argmax(history_R)
        best_params = agent.history_params[best_idx_local]
        best_returns_per_agent.append(history_R[best_idx_local])

        best_traj = envRL.simulate_numba(best_params)
        best_traj = resample_min_jerk(best_traj, N_new=int(CTRL_HZ * sim_duration), duration=sim_duration)
        best_trajs_per_agent.append(best_traj)

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(22, 14))
    ax_traj = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap("nipy_spectral", n_agents)
    for i, traj_best_local in enumerate(best_trajs_per_agent):
        xs, ys, zs = traj_best_local['x'][:, 0], traj_best_local['x'][:, 1], traj_best_local['x'][:, 2]
        ax_traj.plot(xs, ys, zs, color=cmap(i), linewidth=2.5, label=f"Agent {i} best (R={best_returns_per_agent[i]:.3f}")
    for _, ob in enumerate(obstacles):
        c, r = ob['center'], ob['radius']
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
        x = c[0] + r * np.cos(u) * np.sin(v)
        y = c[1] + r * np.sin(u) * np.sin(v)
        z = c[2] + r * np.cos(v)
        ax_traj.plot_surface(x, y, z, color='red', alpha=0.25, linewidth=0)
    set_axes_equal(ax_traj)
    plt.xlabel("Time [s]")
    plt.ylabel("X position [m]")
    plt.title("Best Trajectories per Agent: Close to Simulate")
    plt.legend()
    plt.show()

    # --- Main loop ---
    for step in range(int(sim_duration * CTRL_HZ * 2)):
        for i in range(NUM_DRONES):
            # Get the full state vector for drone i (CtrlAviary provides it internally)
            state_i = env._getDroneStateVector(i)
            if step < int(sim_duration * CTRL_HZ):
                target = best_trajs_per_agent[i]["x"][step][0:3] * scaling + init_xyzs[i]
            else:
                target = goal[0:3] * scaling + init_xyzs[i]
            rpm_i, _, _ = ctrls[i].computeControlFromState(
                control_timestep=dt_ctrl,
                state=state_i,
                target_pos=target,
                target_rpy=np.array([0.0, 0.0, 0.0]),
            )
            action[i, :] = rpm_i

        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(dt_ctrl)

    env.close()

if __name__ == "__main__":
    main()
