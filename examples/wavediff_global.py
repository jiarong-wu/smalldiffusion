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
from smalldiffusion.wavedata import npyDataResized
from smalldiffusion.diffusion import ScheduleLogLinear, samples, my_training_loop

from waveutils import plot_sample

### TODO: refine this to reuse mean and std stats from model reload. And multi-GPU case for computing dataset stats

def main(train_batch_size=1024, epochs=300, sample_batch_size=64, RESUME=False, weights_file=None):
    # Setup
    print(torch.cuda.is_available())
    a = Accelerator(); print(a.state)
    
    # if a.is_main_process: ???
    train_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/train_global/'
    train_file_names = [(f'wave_2011{i:02d}', f'forcing_2011{i:02d}') for i in range(1, 13)]
    train_file_list = [(os.path.join(train_file_path, f'{x}.npy'), 
                        os.path.join(train_file_path, f'{f}.npy')) for x, f in train_file_names]
    # try assigning scale:
    # test.meanx tensor([  2.3457, 160.9466, 191.7947,  44.5279])
    # test.stdx tensor([  1.5441, 120.1778,  95.0668,  15.1788])
    # test.meanf tensor([4.7543e-02, 1.7224e-01, 3.4807e+03, 1.0555e-01])
    # test.stdf tensor([6.4509e+00, 5.3636e+00, 1.7002e+03, 2.8770e-01])
    # train = npyDataResized(
    #     train_file_list,
    #     resize_x=(320,320), resize_f=(320,320), 
    #     maskname=os.path.join(train_file_path, 'mask.npy'),
    #     compute_stats=True
    # )
    train = npyDataResized(
        train_file_list,
        resize_x=(320,320), resize_f=(320,320), 
        maskname=os.path.join(train_file_path, 'mask.npy'),
        compute_stats=False,
        meanx=torch.tensor([0., 0., 0., 0.]),
        stdx=torch.tensor([5., 100., 360., 90.]),
        meanf=torch.tensor([0., 0., 0., 0.]),            
        stdf=torch.tensor([10., 10., 1000., 1.])
    )

    test_file_path = '/global/homes/j/jiarongw/scratch_folder/wave_data/test_global/'
    test_file_names = [('wave_200804', 'forcing_200804')]
    test_file_list = [(os.path.join(test_file_path, f'{x}.npy'), 
                    os.path.join(test_file_path, f'{f}.npy')) for x, f in test_file_names]
    test = npyDataResized(
        test_file_list,
        resize_x=(320,320), resize_f=(320,320), 
        maskname=os.path.join(test_file_path, 'mask.npy'),
        compute_stats=False,
        meanx=train.meanx, stdx=train.stdx, meanf=train.meanf, stdf=train.stdf
    )    

    loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    loader_test = DataLoader(test, batch_size=sample_batch_size, shuffle=True)  # Used for generating samples during training
    loader_test_iter = iter(loader_test)   
    mask_batched = train.mask_resized[None, :, :, :]

    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=80)
    
    # in_ch: number of condition channels + 1 for noise
    # out_ch: number of predicted quantities
    model = Scaled(myUnet)(in_dim=320, in_ch=4, out_ch=4, ch=64, precond_ch=4, 
                           scale=(train.meanx, train.stdx, train.meanf, train.stdf),
                           ch_mult=(1, 2, 2), attn_resolutions=(16,))    
    if RESUME and weights_file is not None:
        model.load_state_dict(torch.load(weights_file, map_location='cpu'))

    # Train
    log_file = open("../run/global/test/loss_log.txt", "w")
    ema = EMA(model.parameters(), decay=0.999)
    ema.to(a.device)
    # Notice it was lr 7e-4 before
    for ns in my_training_loop(loader, model, schedule, epochs=epochs, lr=2e-4, accelerator=a, 
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
            torch.save(model.state_dict(), '../run/global/test/' + 'checkpoint.pth')
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
                print(x.unsqueeze(0).detach().cpu().shape)
                fig = plot_sample(x_.detach().cpu().numpy()[:,::-1],  # truth
                                  x0_.unsqueeze(0).detach().cpu().numpy()[:,:,::-1], # sample
                                  f_.detach().cpu().numpy()[:,::-1])
                fig.savefig('../run/global/test/' + f'sample{i}.png')
            
        
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
    # main(train_batch_size=16, epochs=100, sample_batch_size=2, RESUME=True, weights_file='../run/global/checkpoint_100.pth')
    main(train_batch_size=16, epochs=100, sample_batch_size=2, RESUME=False)    
    