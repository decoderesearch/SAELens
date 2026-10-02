from typing import Any

import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LRScheduler,
)

from sae_lens.training.optim import CoefficientScheduler, get_lr_scheduler

LR = 0.1


@pytest.fixture
def optimizer():
    return Adam([torch.tensor(1.0)], lr=LR)


def step_times(num: int, optimizer: Adam, scheduler: LRScheduler):
    for _ in range(num):
        step(optimizer, scheduler)


def step(optimizer: Adam, scheduler: LRScheduler):
    optimizer.step()
    scheduler.step()


def test_get_scheduler_errors_on_uknown_scheduler(optimizer: Adam):
    with pytest.raises(ValueError, match="Unsupported scheduler: unknown"):
        get_lr_scheduler(
            "unknown",
            optimizer,
            lr=LR,
            training_steps=10,
            warm_up_steps=0,
            decay_steps=0,
            lr_end=0.0,
            num_cycles=1,
        )


def test_get_scheduler_constant(optimizer: Adam):
    scheduler = get_lr_scheduler(
        "constant",
        optimizer,
        lr=LR,
        training_steps=4,
        warm_up_steps=0,
        decay_steps=0,
        lr_end=0.0,
        num_cycles=1,
    )
    assert scheduler.get_last_lr() == [0.1]
    step_times(3, optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]


def test_get_scheduler_constantwithwarmup(optimizer: Adam):
    scheduler = get_lr_scheduler(
        "constant",
        optimizer,
        lr=LR,
        warm_up_steps=2,
        training_steps=4,
        decay_steps=0,
        lr_end=0.0,
        num_cycles=1,
    )
    assert scheduler.get_last_lr() == [pytest.approx(0.05)]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]
    step_times(3, optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]


def test_get_scheduler_linearwarmupdecay(optimizer: Adam):
    scheduler = get_lr_scheduler(
        "constant",
        optimizer,
        lr=LR,
        warm_up_steps=2,
        decay_steps=4,
        training_steps=6,
        lr_end=0.0,
        num_cycles=1,
    )
    # first, ramp up for 2 steps
    assert scheduler.get_last_lr() == [0.05]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]
    step(optimizer, scheduler)
    # next, ramp down for 4 steps
    assert scheduler.get_last_lr() == [0.1]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [pytest.approx(0.075)]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [pytest.approx(0.05)]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [pytest.approx(0.025)]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.0]


def test_get_scheduler_errors_if_lr_end_is_0_and_decay_is_set(optimizer: Adam):
    with pytest.raises(ValueError, match="Cannot have decay_steps with lr_end=0.0"):
        get_lr_scheduler(
            "cosineannealing",
            optimizer,
            lr=LR,
            lr_end=0.0,
            warm_up_steps=0,
            decay_steps=2,
            training_steps=6,
            num_cycles=1,
        )


def test_get_scheduler_cosineannealing(optimizer: Adam):
    scheduler: Any = get_lr_scheduler(
        "cosineannealing",
        optimizer,
        lr=LR,
        training_steps=4,
        lr_end=0.05,
        warm_up_steps=0,
        decay_steps=0,
        num_cycles=1,
    )
    assert len(scheduler._schedulers) == 1
    main_scheduler = scheduler._schedulers[0]
    assert isinstance(main_scheduler, CosineAnnealingLR)
    assert main_scheduler.T_max == 4
    assert main_scheduler.eta_min == 0.05


def test_get_scheduler_cosineannealing_with_warmup_and_decay():
    lr_end = 0.01
    optimizer = Adam([torch.tensor(1.0)], lr=LR)
    scheduler = get_lr_scheduler(
        "cosineannealing",
        optimizer,
        lr=LR,
        warm_up_steps=2,
        training_steps=8,
        decay_steps=2,
        lr_end=lr_end,
        num_cycles=1,
    )
    # first, ramp up for 2 steps
    assert scheduler.get_last_lr() == [0.05]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]
    step(optimizer, scheduler)

    # From here on, it should match CosineAnnealingLR
    new_optimizer = Adam([torch.tensor(1.0)], lr=LR)
    cos_scheduler = CosineAnnealingLR(new_optimizer, T_max=4, eta_min=lr_end)  # type: ignore

    step(optimizer, scheduler)
    step(new_optimizer, cos_scheduler)
    assert scheduler.get_last_lr() == pytest.approx(cos_scheduler.get_last_lr())
    step(optimizer, scheduler)
    step(new_optimizer, cos_scheduler)
    assert scheduler.get_last_lr() == pytest.approx(cos_scheduler.get_last_lr())
    step(optimizer, scheduler)
    step(new_optimizer, cos_scheduler)
    assert scheduler.get_last_lr() == pytest.approx(cos_scheduler.get_last_lr())
    step(optimizer, scheduler)
    step(new_optimizer, cos_scheduler)
    assert scheduler.get_last_lr() == pytest.approx(cos_scheduler.get_last_lr())
    assert scheduler.get_last_lr() == [lr_end]

    # now, decay to 0 in 2 steps
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [pytest.approx(0.005)]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [pytest.approx(0.0)]


def test_get_scheduler_cosineannealingwarmrestarts(optimizer: Adam):
    scheduler: Any = get_lr_scheduler(
        "cosineannealingwarmrestarts",
        optimizer,
        lr=LR,
        training_steps=8,
        lr_end=0.05,
        num_cycles=2,
        warm_up_steps=0,
        decay_steps=0,
    )
    assert len(scheduler._schedulers) == 1
    main_scheduler = scheduler._schedulers[0]
    assert isinstance(main_scheduler, CosineAnnealingWarmRestarts)
    assert main_scheduler.T_0 == 4
    assert main_scheduler.eta_min == 0.05


@pytest.mark.parametrize(
    "steps_taken, expected_value", [(0, 0.0), (4, 2.0), (10, 5.0), (20, 5.0)]
)
def test_CoefficientScheduler_load_state_dict_restores_current_value(
    steps_taken: int, expected_value: float
):
    original = CoefficientScheduler(warm_up_steps=10, final_value=5.0)
    for _ in range(steps_taken):
        original.step()
    assert original.value == pytest.approx(expected_value)

    restored = CoefficientScheduler(warm_up_steps=10, final_value=5.0)
    restored.load_state_dict(original.state_dict())

    assert restored.current_step == original.current_step
    assert restored.value == pytest.approx(original.value)
