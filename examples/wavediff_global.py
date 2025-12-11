import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA
from torchvision import transforms as tf
import torch.distributed as dist

from smalldiffusion.model_unet import myUnet
from smalldiffusion.model import Scaled
from smalldiffusion.data import npyData
from smalldiffusion.diffusion import ScheduleLogLinear, samples, my_training_loop

from waveutils import plot_sample

''' Masked dataset functions and class. '''

class FillNaN(object):
    """ Replace NaNs with zeros (or any value). """
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, tensor):
        return torch.nan_to_num(tensor, nan=self.value)

class Mask(object):
    """ Mask land. Let's say it works only on C*H*W. """
    def __init__(self, mask):
        self.mask = mask
    def __call__(self, tensor):
        return tensor * self.mask.to(tensor.device)
        
class npyDataResized(npyData):
    def __init__(self, *args, resize_x=(320,320), resize_f=(320,320), **kwargs):
        super().__init__(*args, **kwargs)  # call original __init__

        self.mask_original = torch.tensor(self.mask.astype(bool))[None, :, :].float()
        self.mask_resized = tf.Resize(resize_x)(self.mask_original)
        self.original_H = self.X.shape[2]
        self.original_W = self.X.shape[3]
        
        # Patch the transforms
        self.tf_x = tf.Compose([
            FillNaN(0.0),
            tf.Resize(resize_x),
            tf.Normalize(self.meanx.tolist(), self.stdx.tolist()),
            Mask(self.mask_resized)
        ])
        self.tf_f = tf.Compose([
            FillNaN(0.0),
            tf.Resize(resize_f),
            tf.Normalize(self.meanf.tolist(), self.stdf.tolist()),
            Mask(self.mask_resized)
        ])
        # Inverse transforms
        self.inv_tf_x = tf.Compose([
            tf.Resize((self.original_H, self.original_W)),
            tf.Normalize(mean=[0]*len(self.meanx), std=(1/self.stdx).tolist()),
            tf.Normalize(mean=(-(self.meanx/self.stdx)).tolist(), std=[1]*len(self.stdx)),
            Mask(self.mask_original)
        ])
        self.inv_tf_f = tf.Compose([
            tf.Resize((self.original_H, self.original_W)),
            tf.Normalize(mean=[0]*len(self.meanf), std=(1/self.stdf).tolist()),
            tf.Normalize(mean=(-(self.meanf/self.stdf)).tolist(), std=[1]*len(self.stdf)),
            Mask(self.mask_original)
        ])

def main(train_batch_size=1024, epochs=300, sample_batch_size=64, RESUME=False, weights_file=None):
    # Setup
    print(torch.cuda.is_available())
    a = Accelerator(); print(a.state)
    
    train = npyDataResized('../datasets/train_global/wave.npy', '../datasets/train_global/forcing.npy', '../datasets/train_global/mask.npy', 
                        resize_x=(320,320), resize_f=(320,320), compute_stats=True)
    test = npyDataResized('../datasets/test_global/wave.npy', '../datasets/test_global/forcing.npy', '../datasets/test_global/mask.npy', 
                        resize_x=(320,320), resize_f=(320,320), compute_stats=False, meanx=train.meanx, stdx=train.stdx, meanf=train.meanf, stdf=train.stdf)
    loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    loader_test = DataLoader(test, batch_size=sample_batch_size, shuffle=True)  # Used for generating samples during training
    loader_test_iter = iter(loader_test)
    mask_batched = train.mask_resized[None, :, :, :]

    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=80)
    
    # in_ch: number of condition channels + 1 for noise
    # out_ch: number of predicted quantities
    model = Scaled(myUnet)(in_dim=320, in_ch=4, out_ch=4, ch=64, precond_ch=3, ch_mult=(1, 2, 2), attn_resolutions=(16,))
    
    # Load weights if resuming
    if RESUME and weights_file is not None:
        model.load_state_dict(torch.load(weights_file, map_location='cpu'))

    # Train
    log_file = open("../run/global/loss_log.txt", "w")
    ema = EMA(model.parameters(), decay=0.999)
    ema.to(a.device)
    for ns in my_training_loop(loader, model, schedule, epochs=epochs, lr=7e-4, accelerator=a, 
                               conditional=True, mask=mask_batched):
        log_file.write(f"{ns.loss.item():.5}\n")
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()
    log_file.close()
    
    if a.distributed_type != "NO" and dist.is_available() and dist.is_initialized():
        a.wait_for_everyone()

    # Sampling — ONLY RANK 0
    if a.is_main_process:
        with ema.average_parameters():
            # Conditioned sampling
            x, f = next(loader_test_iter)
            *xt, x0 = samples(model, schedule.sample_sigmas(40), gam=1.6, cond=f,
                              batchsize=sample_batch_size, accelerator=a, mask=mask_batched)
            # TODO: write some diagnostic code to visualize samples
            # save_image(img_normalize(make_grid(x0)), 'samples.png')
            for i in range(sample_batch_size):
                x0_ = train.inv_tf_x(x0[i])
                x_ = train.inv_tf_x(x[i])
                f_ = train.inv_tf_f(f[i])
                fig = plot_sample(x0_.detach().cpu().numpy()[:,::-1], x_.detach().cpu().numpy()[:,::-1], f_.detach().cpu().numpy()[:,::-1])
                fig.savefig('../run/global/' + f'sample{i}.png')
            torch.save(model.state_dict(), '../run/global/' + 'checkpoint.pth')
        
    # 3) Make sure everyone waits until the main process finished sampling/saving
    if a.distributed_type != "NO" and dist.is_available() and dist.is_initialized():
        a.wait_for_everyone()

    # 4) Now end training / destroy process group safely
    #    Prefer accelerate's cleanup; call end_training last.
    try:
        a.end_training()
    except Exception:
        # As a fallback, explicitly destroy the group if still initialized
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            
        
if __name__=='__main__':
    main(train_batch_size=16, epochs=200, sample_batch_size=2, RESUME=True, weights_file='../run/global/checkpoint.pth')
    # main(train_batch_size=16, epochs=100, sample_batch_size=2, RESUME=False)    