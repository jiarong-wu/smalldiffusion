import xarray as xr
import numpy as np

def read_save (readpath='/scratch/jw8736/wavecnn/data/', savepath='../datasets/'):
    ### Generate mean data
    ds = xr.open_mfdataset(readpath + '/LOPS_WW3-GLOB-30M_2008*.nc')
    hs_mean = ds.hs.values
    lp_mean = ds.lm.values
    dir_mean = ds.dir.values
    spr_mean = ds.spr.values
    mean = np.stack([hs_mean, lp_mean, dir_mean, spr_mean], axis=0)
    mean = np.swapaxes(mean, 0, 1)
    np.save(savepath + 'wave_mean.npy', mean)

    ### Generate forcing data
    forcing = np.stack([ds.uwnd.values, ds.vwnd.values, ds.dpt.values, ds.ice.values], axis=0)
    np.save(savepath + 'forcing.npy', np.swapaxes(forcing, 0, 1))
    
    ### Generate maxe data
    ### This works with the half-processed data 
    ### e.g. 201101.nc files with hs, lp, dm, dspr variables
    # hs = np.swapaxes(ds.hs.values, 0, 1)
    # lp = np.swapaxes(ds.lp.values, 0, 1)
    # dm = np.swapaxes(ds.dm.values, 0, 1)
    # dspr = np.swapaxes(ds.dspr.values, 0, 1)
    # hs_fill = np.nan_to_num(hs, nan=0.0)   

    # wave_max = np.zeros((hs.shape[0], 4, hs.shape[-2], hs.shape[-1]))
    # for i in range(hs.shape[0]):
    #     max_hs = np.max(hs_fill[i], axis=0)
    #     max_idx  = np.argmax(hs_fill[i], axis=0)
    #     I = np.arange(max_idx.shape[0])[:, None]
    #     J = np.arange(max_idx.shape[1])[None, :]
    #     max_lp = lp[i][max_idx, I, J]
    #     max_dm = dm[i][max_idx, I, J]
    #     max_dspr = dspr[i][max_idx, I, J]
    #     wave_max[i] = np.stack([max_hs, max_lp, max_dm, max_dspr], axis=0)
    # np.save(savepath + 'wave_maxe.npy', wave_max)
    

if __name__=='__main__':
    readpath = '/scratch/jw8736/wave_data/raw/'
    savepath = '/scratch/jw8736/smalldiffusion/datasets/'
    read_save(readpath, savepath)
    # Pick a region
    # wave = np.load('../datasets/wave_mean.npy')
    # forcing = np.load('../datasets/forcing.npy')    
    # np.save('../datasets/train/wave.npy', wave[:401,:,50:114,0:64])
    # np.save('../datasets/train/forcing.npy', forcing[:401,:,50:114,0:64])
    # np.save('../datasets/test/wave.npy', wave[402:,:,50:114,0:64])
    # np.save('../datasets/test/forcing.npy', forcing[402:,:,50:114,0:64])

    # mask = np.load('../datasets/mask.npy')
    # np.save('../datasets/train/mask.npy', mask[50:114,0:64])
    # np.save('../datasets/test/mask.npy', mask[50:114,0:64])