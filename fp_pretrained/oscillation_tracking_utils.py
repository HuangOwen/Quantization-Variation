#!/usr/bin/env python
# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All Rights Reserved.
import torch
from quantization.lsq_layer import  QuantLinear, QuantAct, QuantConv2d, QuantMultiHeadAct, QuantMuitiHeadLinear, QuantMuitiHeadLinear_in, bit_pass


def add_oscillation_trackers(model, *args, **kwarks):
    tracker_dict = {}
    # Add oscillation trackers to all weight quantizers
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear) or isinstance(module, QuantConv2d) or isinstance(module, QuantMuitiHeadLinear) or isinstance(module, QuantMuitiHeadLinear_in):
            if hasattr(module, 'osc_tracker'):
                module.osc_tracker()
            else:
                module.osc_tracker = TrackOscillation(int_fwd=module.to_integer_forward, *args, **kwarks)
            tracker_dict[name + ".weight_quantizer"] = module.osc_tracker
    return tracker_dict


class TrackOscillation:
    """
    This is a wrapper of the int_forward function of a quantizer.
    It tracks the oscillations in integer domain.
    """

    def __init__(self, int_fwd, momentum=0.01, freeze_threshold=0, use_ema_x_int=True):
        self.int_fwd = int_fwd
        self.momentum = momentum

        self.prev_x_int = None
        self.prev_switch_dir = None

        # Statistics to log
        self.ema_oscillation = None
        self.oscillated_sum = None
        self.total_oscillation = None
        self.iters_since_reset = 0

        # Extra variables for weight freezing
        self.freeze_threshold = freeze_threshold  # This should be at least 2-3x the momentum value.
        self.use_ema_x_int = use_ema_x_int
        self.frozen = None
        self.frozen_x_int = None
        self.ema_x_int = None

    def __call__(self, skip_tracking=False, *args, **kwargs):
        x_int = self.int_fwd(*args, **kwargs)

        # Apply weight freezing
        if self.frozen is not None:
            x_int = ~self.frozen * x_int + self.frozen * self.frozen_x_int

        if skip_tracking:
            return x_int

        with torch.no_grad():
            # Check if everything is correctly initialized, otherwise do so
            self.check_init(x_int)

            # detect difference in x_int  NB we round to avoid int inaccuracies
            delta_x_int = torch.round(self.prev_x_int - x_int).detach()  # should be {-1, 0, 1}
            switch_dir = torch.sign(delta_x_int)  # This is {-1, 0, 1} as sign(0) is mapped to 0
            # binary mask for switching
            switched = delta_x_int != 0

            oscillated = (self.prev_switch_dir * switch_dir) == -1
            self.ema_oscillation = (
                self.momentum * oscillated + (1 - self.momentum) * self.ema_oscillation
            )

            # Update prev_switch_dir for the switch variables
            self.prev_switch_dir[switched] = switch_dir[switched]
            self.prev_x_int = x_int
            self.oscillated_sum = oscillated.sum()
            self.total_oscillation += oscillated
            self.iters_since_reset += 1

            # Freeze some weights
            if self.freeze_threshold > 0:
                freeze_weights = self.ema_oscillation > self.freeze_threshold
                self.frozen[freeze_weights] = True  # Set them to frozen
                if self.use_ema_x_int:
                    self.frozen_x_int[freeze_weights] = torch.round(self.ema_x_int[freeze_weights])
                    # Update x_int EMA which can be used for freezing
                    self.ema_x_int = self.momentum * x_int + (1 - self.momentum) * self.ema_x_int
                else:
                    self.frozen_x_int[freeze_weights] = x_int[freeze_weights]

        return x_int

    def check_init(self, x_int):
        if self.prev_x_int is None:
            # Init prev switch dir to 0
            self.prev_switch_dir = torch.zeros_like(x_int)
            self.prev_x_int = x_int.detach()  # Not sure if needed, don't think so
            self.ema_oscillation = torch.zeros_like(x_int)
            self.oscillated_sum = 0
            self.total_oscillation = torch.zeros_like(x_int)
            print("Init tracking", x_int.shape)
        else:
            assert (
                self.prev_x_int.shape == x_int.shape
            ), "Tracking shape does not match current tensor shape."

        # For weight freezing
        if self.frozen is None and self.freeze_threshold > 0:
            self.frozen = torch.zeros_like(x_int, dtype=torch.bool)
            self.frozen_x_int = torch.zeros_like(x_int)
            if self.use_ema_x_int:
                self.ema_x_int = x_int.detach().clone()
            print("Init freezing", x_int.shape)
