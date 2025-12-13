from matplotlib import pyplot as plt

# Plotting sample v.s. truth, and forcing (only wind vectors for now)
# x: sampled wave data, shape (C, H, W)
# x_true: true wave data, shape (C, H, W)
# f: forcing data, shape (C_f, H, W)
def plot_sample(x_true, x, f):
    n_sample = x.shape[0]
    fig, axes = plt.subplots(1+n_sample, 6, figsize=[44, 3*(1+n_sample)], dpi=100)
    titles = ['wave height','wave length','direction','spread','wind u','wind v']
    imgs = [
        axes[0, 0].imshow(x_true[0], vmin=0, vmax=10, cmap='Blues'),
        axes[0, 1].imshow(x_true[1], vmin=0, vmax=300, cmap='Reds'),
        axes[0, 2].imshow(x_true[2], vmin=0, vmax=360, cmap='twilight_r'),
        axes[0, 3].imshow(x_true[3], vmin=0, vmax=90, cmap='Grays'),
        axes[0, 4].imshow(f[0]),
        axes[0, 5].imshow(f[1]),
    ]
    axes[0, 0].set_ylabel(f"Truth", fontsize=14, rotation=0, labelpad=40)
    for j in range(6):
        axes[0,j].set_title(titles[j])
        fig.colorbar(imgs[j], ax=axes[0,j], fraction=0.046, pad=0.04)
        
    for i in range(0, n_sample):
        imgs_sample =[
            axes[i+1, 0].imshow(x[i][0], vmin=0, vmax=10, cmap='Blues'),
            axes[i+1, 1].imshow(x[i][1], vmin=0, vmax=300, cmap='Reds'),
            axes[i+1, 2].imshow(x[i][2], vmin=0, vmax=360, cmap='twilight_r'),
            axes[i+1, 3].imshow(x[i][3], vmin=0, vmax=90, cmap='Grays'),
            axes[i+1, 4].imshow(f[0]),
            axes[i+1, 5].imshow(f[1]),
        ]
        axes[i+1, 0].set_ylabel(f"Sample {i}", fontsize=14, rotation=0, labelpad=40)
        
        for j in range(6):
            fig.colorbar(imgs_sample[j], ax=axes[i+1,j], fraction=0.046, pad=0.04)
    
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
    return fig

def plot_wave(x, label='None'):
    n_sample = x.shape[0]
    fig, axes = plt.subplots(1, 4, figsize=[30, 3], dpi=100)
    titles = ['wave height','wave length','direction','spread']
    imgs = [
        axes[0].imshow(x[0], vmin=0, vmax=10, cmap='Blues'),
        axes[1].imshow(x[1], vmin=0, vmax=300, cmap='Reds'),
        axes[2].imshow(x[2], vmin=0, vmax=360, cmap='twilight_r'),
        axes[3].imshow(x[3], vmin=0, vmax=90, cmap='Grays'),
    ]
    axes[0].set_ylabel(label, fontsize=14, rotation=0, labelpad=40)
    for j in range(4):
        axes[j].set_title(titles[j])
        fig.colorbar(imgs[j], ax=axes[j], fraction=0.046, pad=0.04)
        
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
    return fig
