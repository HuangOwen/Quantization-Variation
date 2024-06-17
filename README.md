# Quantization Variation
Official PyTorch implementation of the anonymous ACM MM Submission "Quantization Variation: A New Perspective on Training Transformers with Low-Bit Precision" 

<div align=center>
<img width=80% src="variation.png"/>
</div>


## Preparation

### Requirements

- PyTorch 1.7.0+ and torchvision 0.8.1+ and pytorch-image-models 0.3.2
```shell
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

### Data and Soft Label

- Install PyTorch and ImageNet dataset following the [official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet). 

- Download the soft label following [FKD](https://github.com/szq0214/FKD) and unzip it. We provide multiple types of [soft labels](http://zhiqiangshen.com/projects/FKD/index.html), and we recommend to use [Marginal Smoothing Top-5 (500-crop)](https://drive.google.com/file/d/14leI6xGfnyxHPsBxo0PpCmOq71gWt008/view?usp=sharing). 

## Run

### Preparing for full-precision baseline model

- Download full-precision pre-trained weights via link provided in [Models](#models).
- (Optional) Train your own full-precision baseline model, please check `./fp_pretrained`.

### Quantization-aware training


- W4A4 DeiT-T Quantization with multi-processing distributed training on a single node with multiple GPUs:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_VVTQ.py \
--dist-url 'tcp://127.0.0.1:10001' \
--dist-backend 'nccl' \
--multiprocessing-distributed --world-size 1 --rank 0 \
--model deit_tiny_patch16_224_quant --batch-size 512 --lr 5e-4 \
--warmup-epochs 0 --min-lr 0 --wbits 4 --abits 4 --reg \
--softlabel_path ./FKD_soft_label_500_crops_marginal_smoothing_k_5 \
--finetune [path to full precision baseline model] \
--save_checkpoint_path ./DeiT-T-4bit --log ./log/DeiT-T-4bit.log\
--data [imagenet-folder with train and val folders]
```

### Evaluation
```
CUDA_VISIBLE_DEVICES=0 python train_VVTQ.py \
--model deit_tiny_patch16_224_quant --batch-size 512 --wbits 4 --abits 4 \
--resume [path to W4A4 DeiT-T ckpt] --evaluate --log ./log/DeiT-T-W4A4.log \
--data [imagenet-folder with train and val folders]
```



