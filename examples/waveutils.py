from matplotlib import pyplot as plt

def plot_sample(x, f):
    n = x.shape[0]
    fig, axes = plt.subplots(n, 6, figsize=[22, 3*n], dpi=100)
    titles = ['wave height','wave length','direction','spread','wind u','wind v']
    for i in range(n):
        imgs = [
            axes[i, 0].imshow(x[i,0]),
            axes[i, 1].imshow(x[i,1]),
            axes[i, 2].imshow(x[i,2]),
            axes[i, 3].imshow(x[i,3]),
            axes[i, 4].imshow(f[i,0]),
            axes[i, 5].imshow(f[i,1]),
        ]
        for j in range(6):
            axes[i,j].set_title(titles[j])
            fig.colorbar(imgs[j], ax=axes[i,j], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig
