from matplotlib import pyplot as plt

# Plotting sample v.s. truth, and forcing (only wind vectors for now)
# x: sampled wave data, shape (C, H, W)
# x_true: true wave data, shape (C, H, W)
# f: forcing data, shape (C_f, H, W)
def plot_sample(x, x_true, f):
    fig, axes = plt.subplots(2, 6, figsize=[44, 6], dpi=100)
    titles = ['wave height','wave length','direction','spread','wind u','wind v']
    imgs = [
        axes[0, 0].imshow(x_true[0], vmin=0, vmax=10, cmap='Blues'),
        axes[0, 1].imshow(x_true[1], vmin=0, vmax=200, cmap='Reds'),
        axes[0, 2].imshow(x_true[2], vmin=0, vmax=360, cmap='twilight_r'),
        axes[0, 3].imshow(x_true[3], vmin=0, vmax=90, cmap='Grays'),
        axes[0, 4].imshow(f[0]),
        axes[0, 5].imshow(f[1]),
    ]
    imgs_sample =[
        axes[1, 0].imshow(x[0], vmin=0, vmax=10, cmap='Blues'),
        axes[1, 1].imshow(x[1], vmin=0, vmax=200, cmap='Reds'),
        axes[1, 2].imshow(x[2], vmin=0, vmax=360, cmap='twilight_r'),
        axes[1, 3].imshow(x[3], vmin=0, vmax=90, cmap='Grays'),
        axes[1, 4].imshow(f[0]),
        axes[1, 5].imshow(f[1]),
    ]
    for j in range(6):
        axes[0,j].set_title(titles[j])
        fig.colorbar(imgs[j], ax=axes[0,j], fraction=0.046, pad=0.04)
        fig.colorbar(imgs_sample[j], ax=axes[1,j], fraction=0.046, pad=0.04)
        
    fig.text(0.02, 0.72, "Truth", va='center', ha='left', fontsize=14)
    fig.text(0.02, 0.28, "Sampled",     va='center', ha='left', fontsize=14)
    
    plt.tight_layout(rect=[0.06, 0, 1, 1])  # leave room on the left for text
    return fig
