import time
import numpy as np
import pybullet as p
import os
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from RL.dmp import MixturePD
from RL.env_reaching import ReachingEnv
from RL.demo_utils import init_from_demo, make_exploration_std, set_axes_equal, make_demo_6D
from RL.multiagent_power_rl import MultiAgentPowerRL
from RL.resample import resample_min_jerk

from MPC.LMPC_solver_obs import LinearMPCController
from MPC.QP_solver import MultiDroneCBFQP
import spatialmath as sm

'''' 
Example of using Multi-Agent Power RL to learn reaching trajectories for multiple drones in a PyBullet environment.
The current script initializes a circular formation of drones, sets up a reaching task with obstacles, and runs the learning algorithm.
The Drones then execute their best learned trajectories in the simulation, with a LMPC controller to ensure safe obstacle avoidance.
'''

def main(plotTraj = False):
    # --- Init Drones, start, and goal ---
    NUM_DRONES = 5
    r = 0.2      # circle radius (meters)
    center = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # circle center
    goal = np.array([0.0, 2.0, 1.0, 0.0, 0.0, 0.0])   # reaching goal

    angles = np.linspace(0, 2*np.pi, NUM_DRONES, endpoint=False)

    init_xyzs = np.zeros((NUM_DRONES, 6))
    init_xyzs[:, 0] = center[0] + r * np.cos(angles)   # x
    init_xyzs[:, 1] = center[1] + r * np.sin(angles)   # y
    init_xyzs[:, 2] = center[2]                        # z

    ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(NUM_DRONES)]
    init_rpys = np.zeros((NUM_DRONES, 3)) # initial roll/pitch/yaw

    # --- Choose obstacles ---
    centerObs = (center[:3] + goal[:3]) / 2
    radiusObs = np.linalg.norm(goal[:3] - center[:3])/6
    obstacles = [
        {'center': centerObs, 'radius': radiusObs},
    ]

    # Simulation/control rates
    CTRL_HZ = 200
    dt_ctrl = 1.0 / CTRL_HZ
    sim_duration = 6.0 # seconds
    traj_duration = 1.5 # seconds

    # --- Init RL ---
    np.random.seed(1)

    # --- Parameters ---
    D, K = 6, 4
    duration, dt = 1.0, 0.05 # DMP duration and timestep, later resampled to CTRL_HZ and desired traj duration, but kept small for faster simulation
    weight_demo, weight_goal = 0.05, 0.95
    weight_jerk, weight_end_vel = 0.10 * dt * dt, 0.8 * dt
    weight_collision, max_distance_penalty = 5.0, radiusObs*2
    n_iterations, rollouts_per_agent = 1200, 8
    n_agents = NUM_DRONES
    goal_start_dist = np.linalg.norm(goal[:3] - center[:3])
    exploration_std = make_exploration_std(D, K, sigma_pos= goal_start_dist, sigma_ori=0.0, sigma_kp=0.05 * goal_start_dist)
    decay = 0.98

    # --- DMP and environment setup ---
    dmp = MixturePD(D=D, K=K, duration=duration, kp_diag=160.0, vel_gain=12.0)
    demo = make_demo_6D(duration=duration, timesteps=int(duration / dt), start=center, goal=goal)
    dmp = init_from_demo(dmp, demo, kp_diag=80.0)

    envRL = ReachingEnv(dmp, dt=dt, obstacles=obstacles, demo_traj=demo, goal=goal)

    # --- Multi-agent RL system ---
    population = MultiAgentPowerRL(
        dmp.get_flat_params(),
        exploration_std=exploration_std,
        n_agents=n_agents,
        reuse_top_n=3,
        diversity_strength = np.mean(exploration_std)
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
                start_x = init_xyzs[agent_id, :],
                w_collision=weight_collision,
                max_distance_penalty=max_distance_penalty,
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
        population.apply_diversity_pressure(exploration_std=exploration_std*0.0)
        population.update_exploration(exploration_std * (decay ** it))
        population.update_diversity_strength(population.diversity_strength * decay)

    # --- Compute best traj/return per agent ---
    best_trajs_per_agent = []
    best_returns_per_agent = []

    for i, agent in enumerate(population.agents):
        if len(agent.history_returns) == 0:
            best_trajs_per_agent.append(None)
            best_returns_per_agent.append(-np.inf)
            continue

        history_R = np.array(agent.history_returns)
        best_idx_local = np.argmax(history_R)
        best_params = agent.history_params[best_idx_local]
        best_returns_per_agent.append(history_R[best_idx_local])

        best_traj = envRL.simulate_numba(best_params, start_x=init_xyzs[i, :])
        best_traj = resample_min_jerk(best_traj, N_new=int(CTRL_HZ * traj_duration), duration=traj_duration)
        best_trajs_per_agent.append(best_traj)
    
    if plotTraj:
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
        
        xs, ys, zs = goal[0], goal[1], goal[2]
        ax_traj.scatter(xs, ys, zs, color='green', s=100, label='Goal')
        for i in range(NUM_DRONES):
            xs, ys, zs = init_xyzs[i, 0], init_xyzs[i, 1], init_xyzs[i, 2]
            ax_traj.scatter(xs, ys, zs, color=cmap(i), s=100)
        set_axes_equal(ax_traj)
        plt.xlabel("Time [s]")
        plt.ylabel("X position [m]")
        plt.title("Best Trajectories per Agent: Close to Simulate")
        plt.legend()
        plt.show()

    # Drone Env
    env = VelocityAviary(
        drone_model=DroneModel.CF2X,
        num_drones=NUM_DRONES,
        initial_xyzs=init_xyzs[:, :3],
        initial_rpys=init_rpys,
        physics=Physics.PYB,
        gui=True,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    # Disable all GUI elements
    p.connect(p.DIRECT, options="--fullscreen")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # Set initial camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=70,
        cameraPitch=-25,
        cameraTargetPosition=centerObs[:3],
    )

    action = np.zeros((NUM_DRONES, 4))

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

    time.sleep(1.0)  # wait for a bit
    # --- Main loop ---
    speed_limit = 5.0 # max LMPC speed
    n = 3
    dt = dt_ctrl * n # LMPC timestep, multiple of control timestep usually
    horizon = 10 # Horizon steps, each of dt seconds
    gamma = 0.05 # weight for control effort in LMPC
    obstacle_margin = 0.15 # margin to consider around obstacles in LMPC, i.e the size of the drone plus some margin
    
    # --- LMPC setup ---, by default uses 6 DoF trajectory (vx, vy, vz, wx, wy, wz), with constraints on each component of the control input
    # The orientation is not used for the Crazyflies, but could be used for other drone models
    lmpc_solver = LinearMPCController(horizon=horizon, dt=dt, gamma = gamma,
                                    u_min=np.array([-speed_limit, -speed_limit, -speed_limit, -speed_limit, -speed_limit, -speed_limit]),
                                    u_max=np.array([ speed_limit,  speed_limit,  speed_limit,  speed_limit,  speed_limit,  speed_limit]))
    Uopt = np.zeros((6 * lmpc_solver.horizon,))
    v_nom = np.zeros((NUM_DRONES, 3))
    cbf_qp_solver = MultiDroneCBFQP(num_drones=NUM_DRONES, dt=dt_ctrl)

    for step in range(int(sim_duration * CTRL_HZ)):
        t0 = time.time()
        for i in range(NUM_DRONES):
            # Get the full state vector for drone i (CtrlAviary provides it internally)
            state_i = env._getDroneStateVector(i)
            position_i = state_i[0:3]
            velocity_i = state_i[10:13]
            omega_i = state_i[13:16]
            xi0 = np.hstack((velocity_i, omega_i)) # initial state for LMPC: current linear and angular velocities
            xi0 = np.clip(xi0, -speed_limit/2, speed_limit/2) # clip to avoid too large initial velocities, happens during inter-collisions for example

            if step < int(traj_duration * CTRL_HZ):
                target = best_trajs_per_agent[i]["x"][step][0:3]
            else:
                target = best_trajs_per_agent[i]["x"][-1][0:3]
            if step < int(traj_duration * CTRL_HZ) - lmpc_solver.horizon:
                traj = best_trajs_per_agent[i]["x"][step:step + lmpc_solver.horizon, 0:3]
            else:
                traj = None

            T_i = sm.SE3.Trans(position_i)
            T_des = sm.SE3.Trans(target)
            
            if step % n == 0: # We solve the LMPC every n control steps, as specified by the architecture
                Uopt, Xopt, poses = lmpc_solver.solve(T_i, T_des, xi0=xi0, obstacles=obstacles, traj=traj, margin=obstacle_margin)
                v_nom[i, :] = Uopt[0:3]

        # CBF-QP to adjust velocities to avoid collisions
        states = [env._getDroneStateVector(j) for j in range(NUM_DRONES)]
        pos = np.array([s[0:3] for s in states])

        v_opt, slack = cbf_qp_solver.solve(
            v_nom=v_nom,
            positions=pos,
            obstacles=obstacles,
            v_max=speed_limit*5,
            d_obs_margin=obstacle_margin, # minimum distance to obstacles
            d_safe=obstacle_margin * 2, # minimum distance between drones
            alpha_obs=10,
            alpha_pair=10,
            use_slack=False, # Optionnaly, use slack variables to always ensure feasibility
            rho_slack=1e4,
        )

        # Set actions
        for i in range(NUM_DRONES):
            v_safe_i = v_opt[i, :]
            speed = np.linalg.norm(v_safe_i)
            direction = v_safe_i / speed if speed > 1e-3 else np.zeros(3)
            action[i, 0:3] = direction
            action[i, 3] = speed

        obs, reward, terminated, truncated, info = env.step(action)
        t1 = time.time()
        elapsed = t1 - t0
        if elapsed < dt_ctrl:
            time.sleep(dt_ctrl - elapsed)
        else:
            time.sleep(dt_ctrl)

    env.close()

if __name__ == "__main__":
    main(plotTraj=False)
