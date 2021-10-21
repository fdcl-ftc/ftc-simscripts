"""Notes
ftc commit: d029bcd
Objective: Modularized main
"""
import numpy as np
import matplotlib.pyplot as plt

import fym
from fym.core import BaseEnv, BaseSystem

import ftc.config
from ftc.models.multicopter import Multicopter
from ftc.faults.actuator import LoE
from ftc.faults.manager import LoEManager
from ftc.agents.switching import Switching
from ftc.agents.lqr import LQRController
from ftc.agents.switching_lqr import LQRLibrary
from ftc.agents.CA import CA, ConstrainedCA
from ftc.plotting import exp_plot

cfg = ftc.config.load()

ftc.config.set({
    "cfg.agents.fdi": {
        "delay": 0.1,
    },

    "cfg.agents.lqr": {

        "LQRGain": {
            "Q": np.diag([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]),
            "R": np.diag([1, 1, 1, 1]),
        },
    },

    "cfg.agents.switching_lqr": {

        "LQRGainList": [
            # No failure
            {
                "Q": np.diag(np.hstack((
                    [10, 10, 10],
                    [1, 1, 1],
                    [100, 100, 100],
                    [1, 1, 1],
                ))),
                "R": np.diag([1, 1, 1, 1, 1, 1]),
            },

            # One failure
            {
                "Q": np.diag(np.hstack((
                    [10, 10, 10],
                    [1, 1, 1],
                    [100, 100, 100],
                    [1, 1, 1],
                ))),
            },

            # Two failures
            {
                "Q": np.diag(np.hstack((
                    [1000, 1000, 1000],
                    [100, 100, 100],
                    [0, 0, 0],
                    [1, 1, 1],
                ))),
            },
        ],
    },

    "cfg.models.multicopter": {
        "init.pos": np.zeros((3, 1)),
    },
})


class ActuatorDynamcs(BaseSystem):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def set_dot(self, rotors, rotors_cmd):
        self.dot = - 1 / self.tau * (rotors - rotors_cmd)


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        init = cfg.models.multicopter.init
        self.plant = Multicopter(init.pos, init.vel, init.quat, init.omega)
        n = self.plant.mixer.B.shape[1]
        self.trim_forces = np.vstack([self.plant.m * self.plant.g, 0, 0, 0])

        # Simulation Settings
        faults = [
            # LoE(time=3, index=0, level=0.),
            # LoE(time=6, index=2, level=0.),
        ]

        ctrls = [LQRLibrary, LQRController]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.sensor_faults = []
        self.fault_manager = LoEManager(faults, no_act=n)

        # Define FDI
        self.fdi = self.fault_manager.fdi

        # Define agents
        self.switching = Switching()
        self.controller1 = ctrls[0](self.plant)
        self.controller2 = ctrls[1](self.plant.Jinv,
                                    self.plant.m,
                                    self.plant.g)
        self.CA = CA(self.plant.mixer.B)
        self.CCA = ConstrainedCA(self.plant.mixer.B)

        self.detection_time = self.fault_manager.fault_times + self.fdi.delay

        # Set references
        pos_des = np.vstack([-1, 1, 2])
        vel_des = np.vstack([0, 0, 0])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        self.ref = np.vstack([pos_des, vel_des, quat_des, omega_des])

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, t, forces, What):
        fault_index = self.fdi.get_index(t)

        if len(fault_index) == 0:
            rotors = np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(forces)
        else:
            rotors = self.CCA.solve_opt(fault_index, forces,
                                        self.plant.rotor_min,
                                        self.plant.rotor_max)
        return rotors

    def get_ref(self, t):
        return self.ref

    def set_dot(self, t):
        mult_states = self.plant.state
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        ref = self.get_ref(t)

        # Set sensor faults
        for sen_fault in self.sensor_faults:
            mult_states = sen_fault(t, mult_states)

        # Controller
        fault_index = self.fdi.get_index(t)
        if self.switching.flag == 1:
            forces = self.controller2.get_FM(mult_states, ref)
            rotors_cmd = self.control_allocation(t, forces, What)
        # elif self.switching.flag == 2:
        #     rotors_cmd = self.controller3.get_rotors(mult_states, ref, fault_index)
        else:
            rotors_cmd = self.controller1.get_rotors(mult_states, ref, fault_index)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        rotors = self.fault_manager.get_faulty_input(t, rotors)

        self.plant.set_dot(t, rotors)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref)


def run(loggerpath):
    env = Env()
    env.logger = fym.Logger(loggerpath)
    env.logger.set_info(cfg=ftc.config.load())

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            env_info = {
                "detection_time": env.detection_time,
                "rotor_min": env.plant.rotor_min,
                "rotor_max": env.plant.rotor_max,
            }
            env.logger.set_info(**env_info)
            break

    env.close()


def exp1(loggerpath):
    run(loggerpath)


if __name__ == "__main__":
    loggerpath = "data.h5"
    exp1(loggerpath)
    exp_plot(loggerpath)
