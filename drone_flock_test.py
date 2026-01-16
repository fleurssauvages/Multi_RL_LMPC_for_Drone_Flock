import time
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

def main():
    NUM_DRONES = 5

    # Initial positions (spread out so they don’t collide at start)
    init_xyzs = np.array([
        [0.0,  0.0, 0.1],
        [0.5,  0.0, 0.1],
        [0.0,  0.5, 0.1],
        [-0.5, 0.0, 0.1],
        [0.0, -0.5, 0.1],
    ])
    init_rpys = np.zeros((NUM_DRONES, 3))

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

    ctrls = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(NUM_DRONES)]

    # Example: hold a formation at z=1.0
    targets = np.array([
        [ 0.0,  0.0, 1.0],
        [ 0.5,  0.0, 1.0],
        [ 0.0,  0.5, 1.0],
        [-0.5,  0.0, 1.0],
        [ 0.0, -0.5, 1.0],
    ])

    env.reset()

    for step in range(10_000):
        action = np.zeros((NUM_DRONES, 4))

        for i in range(NUM_DRONES):
            # Get the full state vector for drone i (CtrlAviary provides it internally)
            state_i = env._getDroneStateVector(i)

            rpm_i, _, _ = ctrls[i].computeControlFromState(
                control_timestep=dt_ctrl,
                state=state_i,
                target_pos=targets[i],
                target_rpy=np.array([0.0, 0.0, 0.0]),
            )
            action[i, :] = rpm_i

        obs, reward, terminated, truncated, info = env.step(action)

        # Optional: real-time-ish
        time.sleep(dt_ctrl)

    env.close()

if __name__ == "__main__":
    main()
