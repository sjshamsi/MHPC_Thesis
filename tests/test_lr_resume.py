"""Resuming from a checkpoint must not change the learning-rate schedule.

Trains the same tiny model twice:
  A: 3 epochs uninterrupted.
  B: 1 epoch, then a second main() call that auto-resumes from B/last and trains to epoch 3.
The learning rate saved in each epoch's checkpoint must be the same in A and B.

The scheduler steps at the end of each epoch, before the checkpoint is saved, so a
checkpoint already holds the next epoch's learning rate. On resume, train.py builds a new
scheduler, whose constructor takes one more step.

`InverseSqrtLinearRamps` computes each learning rate from the current one, so that extra
step applies the epoch's factor a secondtime.
`InverseSqrtLinearWarmupSqrtCooldown` computes it from the epoch number alone, so it is the
control case and passes either way.

All epochs stay inside the 10-epoch warmup of both schedulers: the first half of run B uses
max_epoch=1 while run A uses 3, and warmup is the only phase that does not depend on
max_epoch.

For speed, we use model=debug, data=test (2-trajectory dummy dataset) whixh come from
conftest.py's default_conf and are already tiny. SPEED_OVERRIDES cut batches per epoch and
validation length. None of them change the learning rate, which the scheduler
only updates once per epoch (trainer.lr_scheduler_per_step=False).
"""

import os.path as osp
import pathlib

import pytest
import torch
from hydra import compose, initialize

from walrus.train import CONFIG_DIR, CONFIG_NAME, main

SCHEDULERS = [
    pytest.param({"lr_scheduler": "inv_sqrt_w_linear_ramps"}, id="linear_ramps"),
    pytest.param({"lr_scheduler": "inv_sqrt_w_sqrt_ramps"}, id="sqrt_ramps"),
]

SPEED_OVERRIDES = [
    "data.module_parameters.max_samples=2",     # 2 micro-batches per epoch instead of 8
    "trainer.grad_acc_steps=1",                 # so those 2 micro-batches give 2 optimizer steps (default 4: none)
    "data_workers=0",                           # load batches in the main process: no worker start-up
    "++trainer.debug_mode=True",                # cap validation/test at short_validation_length batches
    "trainer.short_validation_length=1",
    "trainer.max_rollout_steps=2",
    "++trainer.skip_spectral_metrics=True",
]


def run(overrides):
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    with initialize(config_path=str(cfg_dir)):
        main(compose(config_name=CONFIG_NAME, overrides=overrides))


def saved_lrs(step_dir: pathlib.Path):
    """Learning rate of each optimizer param group, as saved in a local checkpoint."""
    sd = torch.load(step_dir / "full_checkpoint.pt", weights_only=False)
    return [group["lr"] for group in sd["app"]["optimizer"]["param_groups"]]


@pytest.mark.parametrize("conf", SCHEDULERS, indirect=True)
def test_resume_keeps_lr_schedule(tmp_path: pathlib.Path, conf):
    # Without a cap, torch starts one thread per core of the (large) login node, and their
    # combined CPU time trips the login node's 600 CPU-second per-process limit.
    torch.set_num_threads(2)
    base = conf + [
        "+model.include_d=[2]",
        "checkpoint.checkpoint_frequency=1",
        *SPEED_OVERRIDES,
    ]
    dir_a, dir_b = tmp_path / "A", tmp_path / "B"

    # A: uninterrupted
    run(base + [f"checkpoint.save_dir={dir_a}", "trainer.max_epoch=3"])
    # B: stop after epoch 1, then resume to epoch 3
    run(base + [f"checkpoint.save_dir={dir_b}", "trainer.max_epoch=1"])
    run(base + [f"checkpoint.save_dir={dir_b}", "trainer.max_epoch=3"])

    for step in ["step_1", "step_2", "step_3"]:
        lr_a, lr_b = saved_lrs(dir_a / step), saved_lrs(dir_b / step)
        ratios = [b / a for a, b in zip(lr_a, lr_b)]
        assert lr_b == pytest.approx(lr_a), (
            f"{step}: resumed lr {lr_b} != uninterrupted lr {lr_a} (ratio {ratios})"
        )
