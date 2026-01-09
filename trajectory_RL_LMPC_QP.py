import time
import numpy as np
import pybullet as p

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

    env = VelocityAviary(
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
    n_agents = NUM_DRONES
    exploration_std = make_exploration_std(D, K, sigma_pos=0.05, sigma_ori=0.1, sigma_kp=0.02)
    decay = 0.98
    scaling = 0.5

    # --- DMP and environment setup ---
    dmp = MixturePD(D=D, K=K, duration=duration, kp_diag=160.0, vel_gain=12.0)
    demo = make_demo_6D(duration=duration, timesteps=int(duration / dt), curvature=0.05, start=center, goal=goal, normalize=scaling)
    scaling = np.linalg.norm(goal[:3] - center[:3]) / scaling
    dmp = init_from_demo(dmp, demo, kp_diag=80.0)
    goal = demo["x"][-1]

    obstaclesRL = [
        {'center': (centerObs - center[:3]) / scaling, 'radius': radiusObs / scaling},
    ]
    envRL = ReachingEnv(dmp, dt=dt, obstacles=obstaclesRL, demo_traj=demo, goal=goal)

    # --- Multi-agent RL system ---
    population = MultiAgentPowerRL(
        dmp.get_flat_params(),
        exploration_std=exploration_std,
        n_agents=n_agents,
        reuse_top_n=3,
        diversity_strength=0.1 * np.mean(exploration_std)
    )
    # --- Main loop RL ---
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

    for i, agent in enumerate(population.agents):
        if len(agent.history_returns) == 0:
            best_trajs_per_agent.append(None)
            best_returns_per_agent.append(-np.inf)
            continue

        history_R = np.array(agent.history_returns)
        best_idx_local = np.argmax(history_R)
        best_params = agent.history_params[best_idx_local]
        best_returns_per_agent.append(history_R[best_idx_local])

        best_traj = envRL.simulate_numba(best_params)
        best_traj['x'] = best_traj['x'][:, 0:3] * scaling + center[:3]

        # Plot trajectory
        for k in range(len(best_traj['x'])-1):
            t1 = best_traj['x'][k, :3]
            t2 = best_traj['x'][k+1, :3]
            p.addUserDebugLine(t1, t2, [0, 1, 0], lineWidth=3)

        best_traj = resample_min_jerk(best_traj, N_new=int(CTRL_HZ * sim_duration), duration=sim_duration)
        best_trajs_per_agent.append(best_traj)

    # --- Main loop ---
    speed_limit = env.SPEED_LIMIT * 10
    lmpc_solver = LinearMPCController(horizon=5, dt=dt_ctrl, gamma = 0.02,
                                    u_min=np.array([-speed_limit*10, -speed_limit*10, -speed_limit*10, -speed_limit*10, -speed_limit*10, -speed_limit*10]),
                                    u_max=np.array([ speed_limit*10,  speed_limit*10,  speed_limit*10,  speed_limit*10,  speed_limit*10,  speed_limit*10]))
    cbf_qp_solver = MultiDroneCBFQP(num_drones=NUM_DRONES, dt=dt_ctrl)
    
    Uopt = np.zeros((6 * lmpc_solver.horizon,))
    time.sleep(5.0)  # wait before starting main loop
    for step in range(int(sim_duration * CTRL_HZ * 2)):
        v_lmpc = np.zeros((NUM_DRONES, 3))

        # LMPC
        for i in range(NUM_DRONES):
            state_i = env._getDroneStateVector(i)
            position_i = state_i[0:3]
            velocity_i = state_i[10:13]
            omega_i = state_i[13:16]
            xi0 = np.hstack((velocity_i, omega_i))
            xi0 = np.clip(xi0, -speed_limit/2, speed_limit/2)

            if step < int(sim_duration * CTRL_HZ):
                target = best_trajs_per_agent[i]["x"][step][0:3]
            else:
                target = best_trajs_per_agent[i]["x"][-1][0:3]
            if step < int(sim_duration * CTRL_HZ) - lmpc_solver.horizon:
                traj = best_trajs_per_agent[i]["x"][step:step + lmpc_solver.horizon, 0:3]
            else:
                traj = None

            T_i = sm.SE3.Trans(position_i)
            T_des = sm.SE3.Trans(target)

            Uopt, Xopt, poses = lmpc_solver.solve(T_i, T_des, xi0=xi0, obstacles=obstacles, traj=traj, margin=0.1)
            v_cart = Uopt[0:3]
            v_lmpc[i, :] = v_cart
        
        # CBF-QP to adjust velocities to avoid collisions
        states = [env._getDroneStateVector(j) for j in range(NUM_DRONES)]
        pos = np.array([s[0:3] for s in states])
        action = np.zeros((NUM_DRONES, 4), dtype=np.float32)

        v_opt, slack = cbf_qp_solver.solve(
            v_nom=v_lmpc,
            positions=pos,
            obstacles=[],
            v_max=speed_limit*10,
            d_obs_margin=0.15,
            d_safe=0.15,
            alpha_obs=10,
            alpha_pair=10,
            use_slack=True,
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
        time.sleep(dt_ctrl)

    env.close()

if __name__ == "__main__":
    main()