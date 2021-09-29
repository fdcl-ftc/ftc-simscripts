from pathlib import Path
import os
import sys
import time
import itertools
from concurrent.futures import ProcessPoolExecutor
import tqdm
import numpy as np
from loguru import logger

import fym

import ftc.config
from ftc.evaluate.evaluate import calculate_recovery_rate
from ftc.plotting import exp_plot


# cfg = ftc.config.load()
ftc.config.set({
    "path.run": Path("data", "run"),
})


logger.remove()
logger.add(sys.stderr, filter={
    "ftc.agents.switching": "ERROR",
    "": "INFO",
})


def sim(i, initial, Env, cfg):
    loggerpath = Path(cfg.path.run, f"env-{i:03d}.h5")
    env = Env(initial)
    env.logger = fym.Logger(loggerpath)
    env.reset()

    while True:
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

    data, info = fym.load(loggerpath, with_info=True)
    time_index = data["t"] > cfg.env.kwargs.max_t - cfg.evaluation.cuttime
    alt_error = cfg.ref.pos[2] - data["x"]["pos"][time_index, 2, 0]
    fym.parser.update(info, dict(alt_error=np.mean(alt_error)))
    fym.save(loggerpath, data, info=info)


def sim_parallel(initial_set, Env, cfg):
    # Initialize concurrent
    cpu_workers = os.cpu_count()
    max_workers = int(cfg.parallel.max_workers or cpu_workers)
    assert max_workers <= os.cpu_count(), \
        f"workers should be less than {cpu_workers}"
    logger.info(f"Sample with {max_workers} workers ...")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers) as p:
        list(tqdm.tqdm(
            p.map(sim, range(cfg.episode.N), initial_set, itertools.repeat(Env), itertools.repeat(cfg)),
            total=cfg.episode.N
        ))

def evaluate(cfg):
    alt_errors = []
    for i in range(cfg.episode.N):
        loggerpath = Path(cfg.path.run, f"env-{i:03d}.h5")
        data, info = fym.load(loggerpath, with_info=True)
        alt_errors = np.append(alt_errors, info["alt_error"])
    recovery_rate = calculate_recovery_rate(alt_errors, threshold=0.5)
    print(recovery_rate)
    logger.info(f"Recovery rate is {recovery_rate:.3f}.")
    exp_plot(loggerpath)
