"""Notes
ftc commit : 2a153c8
Algorithm: Switching LQR
Objective: success rate evaluation using parallel simulation
"""
import numpy as np

import fym
from fym.utils.rot import angle2quat

import ftc.config
from ftc.models.multicopter import Multicopter
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager
from ftc.agents.switching import LQRLibrary

from src.sim import sim_parallel, evaluate

cfg = ftc.config.load()

print("Change `cfg.episode.N`")
cfg.episode.N = 1  # TODO: only for debugging


class Env(fym.BaseEnv):
    def __init__(self, initial):
        super().__init__(**fym.parser.decode(cfg.env.kwargs))
        pos, vel, angle, omega = initial
        pos[2] = pos[2] - 10
        quat = angle2quat(*angle.ravel()[::-1])
        self.plant = Multicopter(pos, vel, quat, omega)
        self.trim_forces = np.vstack([self.plant.m * self.plant.g, 0, 0, 0])
        n = self.plant.mixer.B.shape[1]

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager([
            LoE(time=3, index=0, level=0.),  # scenario a
            # LoE(time=6, index=2, level=0.),  # scenario b
        ], no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.controller = LQRLibrary(self.plant)

        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

        # Set references
        pos_des = cfg.ref.pos
        vel_des = np.vstack([0, 0, 0])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        self.ref = np.vstack([pos_des, vel_des, quat_des, omega_des])

    def step(self):
        *_, done = self.update()
        return done

    def get_ref(self, t):
        return self.ref

    def set_dot(self, t):
        mult_states = self.plant.state
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        ref = self.get_ref(t)

        # Controller
        fault_index = self.fdi.get_index(t)
        rotors_cmd = self.controller.get_rotors(mult_states, ref, fault_index)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

        self.plant.set_dot(t, rotors)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref)


if __name__ == "__main__":
    # Sampling initial conditions
    # TODO: save and load the initial conditions to improve rng-stable simulation
    np.random.seed(0)
    pos = np.random.uniform(*cfg.episode.range.pos, size=(cfg.episode.N, 3, 1))
    vel = np.random.uniform(*cfg.episode.range.vel, size=(cfg.episode.N, 3, 1))
    angle = np.random.uniform(*cfg.episode.range.angle, size=(cfg.episode.N, 3, 1))
    omega = np.random.uniform(*cfg.episode.range.omega, size=(cfg.episode.N, 3, 1))
    initial_set = np.stack((pos, vel, angle, omega), axis=1)
    # TODO: make Envs modularised
    sim_parallel(initial_set, Env, cfg)
    evaluate(cfg)
