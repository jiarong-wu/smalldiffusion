import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage as EMA

from smalldiffusion.model_unet import myUnet
from smalldiffusion.model import Scaled
from smalldiffusion.data import npyData
from smalldiffusion.diffusion import ScheduleLogLinear, samples, my_training_loop

from waveutils import plot_sample


def main(train_batch_size=1024, epochs=300, sample_batch_size=64):
    # Setup
    a = Accelerator()
    train = npyData('train/wave.npy', 'train/forcing.npy', 'train/mask.npy', compute_stats=True)
    test = npyData('test/wave.npy', 'test/forcing.npy', 'test/mask.npy', compute_stats=False,
                    meanx=train.meanx, stdx=train.stdx, meanf=train.meanf, stdf=train.stdf)
    loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    # Used for generating samples during training
    loader_test = DataLoader(test, batch_size=sample_batch_size, shuffle=True)
    loader_test_iter = iter(loader_test)
    
    schedule = ScheduleLogLinear(sigma_min=0.01, sigma_max=20, N=800)
    model = Scaled(myUnet)(in_dim=64, in_ch=4, out_ch=4, ch=64, precond_ch=3, ch_mult=(1, 2, 2), attn_resolutions=(16,))

    # Train
    ema = EMA(model.parameters(), decay=0.999)
    ema.to(a.device)
    for ns in my_training_loop(loader, model, schedule, epochs=epochs, lr=7e-4, accelerator=a):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        ema.update()

    # Sample
    with ema.average_parameters():
        # Conditioned sampling
        x, f = next(loader_test_iter)
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6, cond=f,
                          batchsize=sample_batch_size, accelerator=a)
        # TODO: write some diagnostic code to visualize samples
        # save_image(img_normalize(make_grid(x0)), 'samples.png')
        x0_ = train.inv_tf_x(x0)
        f_ = train.inv_tf_f(f)
        fig = plot_sample(x0_.detach().cpu().numpy(), f_.detach().cpu().numpy())
        fig.savefig('results/' + 'sample.png')
        torch.save(model.state_dict(), 'results/' + 'checkpoint.pth')
    

if __name__=='__main__':
    main(train_batch_size=8, epochs=10, sample_batch_size=2)