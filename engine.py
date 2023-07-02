# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import numpy 
numpy.set_printoptions(threshold=1e-6)
import torch
import utils
from quantization.lsq_layer import QuantAct, QuantConv2d, QuantLinear, QuantMultiHeadAct, QuantMuitiHeadLinear, QuantMuitiHeadLinear_in


@torch.no_grad()
def initialize_quantization(data_loader, model, device, output_dir, sample_iters=5):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Initialization:'
    if utils.is_main_process():
        with (output_dir / "scales.txt").open("w") as f:
            f.write("weight scales:\n")
            for name, m in model.named_modules():
                if (isinstance(m, QuantLinear) or isinstance(m, QuantConv2d) or isinstance(m, QuantMuitiHeadLinear) or isinstance(m, QuantMuitiHeadLinear_in)) and m.alpha is not None:
                    print(f"initialize the weight scale for module {name}")
                    m.initialize_scale(device)
                    f.write(name + ': ' + str(m.alpha.data) + '\n')

            # switch to evaluation mode
            model.eval()
            f.write("activation scales:\n")
            n = 0
            for images, target in metric_logger.log_every(data_loader, 1, header):
                n += 1
                if n > sample_iters:
                    break
                images = images.to(device, non_blocking=True)
                # compute output
                # with torch.cuda.amp.autocast():
                output = model(images)
            '''
            for name, m in model.named_modules():
                if (isinstance(m, QuantAct) or isinstance(m, QuantMultiHeadAct)) and m.alpha is not None:
                    print(f"initialize the activation scale for module {name}")
                    m.initialize_scale_offset(device)
                    f.write(name + ': ' + str(m.alpha.data) + '\n')
                    if m.offset:
                        f.write("offset" + ': ' + str(m.beta.data) + '\n')
            '''
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return
