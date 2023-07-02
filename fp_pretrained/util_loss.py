# Code is modified from MEAL (https://arxiv.org/abs/1812.02425) and Label Refinery (https://arxiv.org/abs/1805.02641).

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.nn.modules import loss
from quantization.lsq_layer import  QuantLinear, QuantAct, QuantConv2d, QuantMultiHeadAct, QuantMuitiHeadLinear, QuantMuitiHeadLinear_in, bit_pass


class DistributionLoss(loss._Loss):
    """The KL-Divergence loss for the binary student model and real teacher output.

    output must be a pair of (model_output, real_output), both NxC tensors.
    The rows of real_output must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, model_output, real_output):

        self.size_average = True

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        if real_output.requires_grad:
            raise ValueError("real network output should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        real_output_soft = F.softmax(real_output, dim=1)
        del model_output, real_output

        # Loss is -dot(model_output_log_prob, real_output). Prepare tensors
        # for batch matrix multiplicatio
        real_output_soft = real_output_soft.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(real_output_soft, model_output_log_prob)
        if self.size_average:
             cross_entropy_loss = cross_entropy_loss.mean()
        else:
             cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        # model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss

class CosineTempDecay:
    def __init__(self, t_max, temp_range=(20.0, 2.0), rel_decay_start=0):
        self.t_max = t_max
        self.start_temp, self.end_temp = temp_range
        self.decay_start = rel_decay_start * t_max

    def __call__(self, t):
        if t < self.decay_start:
            return self.start_temp

        rel_t = (t - self.decay_start) / (self.t_max - self.decay_start)
        return self.end_temp + 0.5 * (self.start_temp - self.end_temp) * (1 + np.cos(rel_t * np.pi))

class BinReg(nn.Module):
    def __init__(self, lmbda):
        super(BinReg, self).__init__()
        self.lmbda = lmbda
    
    def forward(self, model):
        regularization_term = 0        
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear) or isinstance(module, QuantConv2d) or isinstance(module, QuantMuitiHeadLinear) or isinstance(module, QuantMuitiHeadLinear_in):
                weight_q, weight_fp_clip = module.get_quant_weight()
                regularization_term += self.dampening_loss(weight_fp_clip, weight_q, "sum")
        return self.lmbda * regularization_term

    def dampening_loss(self, weight, weight_q, aggregation="sum"):
        # L &= (s*w_{int} - w)^2
        # We also need to add clipping for both cases, we can do so by using the forward
        w_q = weight_q.detach()  # this is also clipped and our target
        loss = (w_q - weight) ** 2
        if aggregation == "sum":
            return loss.sum()
        elif aggregation == "mean":
            return loss.mean()
        elif aggregation == "kernel_mean":
            return loss.sum(0).mean()
        else:
            raise ValueError(f"Aggregation method '{aggregation}' not implemented.")


class BinRegold(nn.Module):
    def __init__(self, lmbda):
        super(BinReg, self).__init__()
        self.lmbda = lmbda
   
    def forward(self, w_list, bitw_list):

        regularization_term = 0        
        for i in range(len(w_list)):
            nbit_w = bitw_list[i]
            weight = w_list[i]
    
            scale = 2 ** nbit_w - 1
            weight = torch.autograd.Variable(weight)
            weight = torch.tanh(weight)
            weight = weight / (2 * torch.max(torch.abs(weight))) + 0.5
            weight_int = torch.round(weight * scale)

            for quant_bin in range(scale):
                weight_idx_quant_bin = weight_int.eq(quant_bin)
                weight_quant_bin = weight_int[weight_idx_quant_bin] / scale
                if weight_quant_bin.numel() == 0:
                    continue
                w_mean = torch.mean(weight_quant_bin)
                regularization_term += torch.mean(torch.pow(w_mean - weight_quant_bin, 2)) # variance formula: 1/N * sum(squared error)

                real_weight_quant_bin = weight[weight_idx_quant_bin]
                if torch.numel(real_weight_quant_bin) > 1:
                    regularization_term += torch.var(real_weight_quant_bin)

        return self.lmbda * regularization_term