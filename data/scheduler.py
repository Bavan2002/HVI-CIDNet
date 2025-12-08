# ===== Learning Rate Schedulers =====
# Custom LR schedulers for training: warmup + cosine annealing with restarts

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math


# ===== Gradual Warmup Scheduler =====
class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up (increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer: Wrapped optimizer.
        multiplier: Target LR = base_lr * multiplier. If =1.0, LR goes from 0 to base_lr.
        total_epoch: Warmup duration - target LR is reached at this epoch.
        after_scheduler: Scheduler to use after warmup (e.g., CosineAnnealing).
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False  # True when warmup is complete
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        """Compute current learning rate based on warmup progress."""
        # After warmup: delegate to after_scheduler
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        # During warmup: linear increase from 0 (or 1) to target
        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        """Handle ReduceLROnPlateau which requires metrics for stepping."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau called at end of epoch
        if self.last_epoch <= self.total_epoch:
            # Still in warmup phase
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            # After warmup: use ReduceLROnPlateau
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        """Step the scheduler (called once per epoch)."""
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def state_dict(self):
        """Return state dict that properly handles nested after_scheduler."""
        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "after_scheduler")
        }
        if self.after_scheduler is not None:
            state["after_scheduler_state"] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Load state dict and restore after_scheduler state properly.
        Handles both old format (after_scheduler object) and new format (state dict).
        """
        # Handle NEW format: after_scheduler_state dict
        after_scheduler_state = state_dict.pop("after_scheduler_state", None)

        # Handle OLD format: after_scheduler object (from old checkpoints)
        old_after_scheduler = state_dict.pop("after_scheduler", None)

        # Update our state (excluding after_scheduler which we handle separately)
        self.__dict__.update(state_dict)

        if after_scheduler_state is not None and self.after_scheduler is not None:
            # NEW format: load state dict into our after_scheduler
            self.after_scheduler.load_state_dict(after_scheduler_state)
        elif old_after_scheduler is not None and self.after_scheduler is not None:
            # OLD format: extract state from the old object
            self.after_scheduler.load_state_dict(old_after_scheduler.state_dict())

        # Apply the restored LR to the optimizer
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


# ===== Helper Function =====
def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.
    Returns index of the right-closest number in cumulative_period.

    Example: cumulative_period = [100, 200, 300, 400]
      iteration=50  -> returns 0
      iteration=210 -> returns 2
      iteration=300 -> returns 2
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


# ===== Cyclic Cosine Annealing with Restarts =====
class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """Cosine annealing with restarts - CYCLIC version with different eta_mins per cycle.

    Example config:
        periods = [10, 10, 10, 10]
        restart_weights = [1, 0.5, 0.5, 0.5]
        eta_mins = [1e-7, 1e-7, 1e-7, 1e-7]

    Four cycles of 10 epochs each. At epochs 10, 20, 30, scheduler restarts
    with the corresponding restart_weight scaling the base LR.
    """

    def __init__(
        self, optimizer, periods, restart_weights=(1,), eta_mins=(0,), last_epoch=-1
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins  # Minimum LR per cycle
        assert len(self.periods) == len(self.restart_weights), (
            "periods and restart_weights should have the same length."
        )
        # Cumulative periods: [10, 20, 30, 40] for periods [10, 10, 10, 10]
        self.cumulative_period = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute LR using cosine annealing within current cycle."""
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        # Cosine annealing formula: eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * t/T))
        return [
            eta_min
            + current_weight
            * 0.5
            * (base_lr - eta_min)
            * (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - nearest_restart) / current_period)
                )
            )
            for base_lr in self.base_lrs
        ]


# ===== Cosine Annealing with Restarts =====
class CosineAnnealingRestartLR(_LRScheduler):
    """Cosine annealing with restarts - single eta_min for all cycles.

    Example config:
        periods = [10, 10, 10, 10]
        restart_weights = [1, 0.5, 0.5, 0.5]
        eta_min = 1e-7

    Four cycles of 10 epochs each. At epochs 10, 20, 30, scheduler restarts
    with the corresponding restart_weight scaling the base LR.
    """

    def __init__(
        self, optimizer, periods, restart_weights=(1,), eta_min=0, last_epoch=-1
    ):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min  # Single minimum LR for all cycles
        assert len(self.periods) == len(self.restart_weights), (
            "periods and restart_weights should have the same length."
        )
        self.cumulative_period = [
            sum(self.periods[0 : i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute LR using cosine annealing within current cycle."""
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min
            + current_weight
            * 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi * ((self.last_epoch - nearest_restart) / current_period)
                )
            )
            for base_lr in self.base_lrs
        ]
