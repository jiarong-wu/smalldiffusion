import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as tf
from pathlib import Path

### Suggested multi file data class by Claude
class MultiFileNpyData(Dataset):
    def __init__(self, 
        file_list,  # List of tuples: [(Xname, Fname), ...] for each month
        landmaskname=None, use_icymask=True,
        compute_stats=True,
        meanx=None, stdx=None, meanf=None, stdf=None
    ):
        """
        Args:
            file_list: List of tuples, each containing (X_filepath, F_filepath) for one month
            maskname: Path to mask file (static mask only)
            compute_stats: Whether to compute mean/std from data
            meanx, stdx, meanf, stdf: Precomputed statistics (optional)
        """
        super().__init__()
        
        # Load all files with memory mapping
        self.X_files = []
        self.F_files = []
        self.file_lengths = []
        self.cumulative_lengths = [0]
        
        for x_path, f_path in file_list:
            x_mmap = np.load(x_path, mmap_mode="r")
            f_mmap = np.load(f_path, mmap_mode="r")
            
            self.X_files.append(x_mmap)
            self.F_files.append(f_mmap[:, 0:3])  # Load U, V, and icymask
            
            file_len = len(x_mmap) - 1  # -1 to prevent last sample
            self.file_lengths.append(file_len)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + file_len)
        
        self.total_length = self.cumulative_lengths[-1]
        
        # Load mask (fixed land and from mask file)
        # Used for data transform
        if landmaskname != None:
            self.landmask = np.load(landmaskname)
        # Optional: use icymask (time dependent and frm F_files) for e.g. loss
        self.use_icymask = use_icymask

        # Compute or load statistics
        if compute_stats:
            print("Computing dataset mean and std across all files...")
            meanx, stdx, meanf, stdf = self._compute_mean_std()
            meanf[2] = 0.0  # icymask not normalized
            stdf[2] = 1.0  # icymask not normalized
        else:
            assert meanx is not None and stdx is not None, "Must provide meanx/stdx when compute_stats=False"
            assert meanf is not None and stdf is not None, "Must provide meanf/stdf when compute_stats=False"
        
        # Store as tensors
        self.meanx = torch.as_tensor(meanx, dtype=torch.float32)
        self.stdx = torch.as_tensor(stdx, dtype=torch.float32)
        self.meanf = torch.as_tensor(meanf, dtype=torch.float32)
        self.stdf = torch.as_tensor(stdf, dtype=torch.float32)
        
        # Transforms
        self.tf_x = tf.Compose([
            tf.Normalize(self.meanx.tolist(), self.stdx.tolist()),
        ])
        self.tf_f = tf.Compose([    
            tf.Normalize(self.meanf.tolist(), self.stdf.tolist()), 
        ])    
        self.inv_tf_x = tf.Compose([
            tf.Normalize(mean=[0]*len(self.meanx), std=(1/self.stdx).tolist()),
            tf.Normalize(mean=(-(self.meanx/self.stdx)).tolist(), std=[1]*len(self.stdx))
        ])
        self.inv_tf_f = tf.Compose([    
            tf.Normalize(mean=[0]*len(self.meanf), std=(1/self.stdf).tolist()),
            tf.Normalize(mean=(-(self.meanf/self.stdf)).tolist(), std=[1]*len(self.stdf))
        ])

    def _get_file_and_local_idx(self, global_idx):
        """Convert global index to (file_index, local_index) tuple."""
        for file_idx in range(len(self.file_lengths)):
            if global_idx < self.cumulative_lengths[file_idx + 1]:
                local_idx = global_idx - self.cumulative_lengths[file_idx]
                return file_idx, local_idx
        raise IndexError(f"Index {global_idx} out of range")

    def __len__(self):
        return self.total_length
    
    def _compute_mean_std(self):
        """Compute channel-wise mean and std across all files."""           
        # Get number of channels from first file
        n_channels_x = self.X_files[0].shape[1]
        n_channels_f = self.F_files[0].shape[1]
        
        # Initialize accumulators
        sum_x = np.zeros(n_channels_x)
        sum_f = np.zeros(n_channels_f)
        sum_sq_x = np.zeros(n_channels_x)
        sum_sq_f = np.zeros(n_channels_f)
        total_pixels = 0
        
        # Accumulate statistics from each file
        for x_file, f_file in zip(self.X_files, self.F_files):
            if self.use_icymask:
                mask = f_file[:, 2, :, :]  
                mask_broadcast = mask[:, None, :, :].astype(bool)   
                total_pixels += mask_broadcast.sum()
                # print(total_pixels)
            else:
                mask = self.landmask.astype(bool)
                mask_broadcast = mask[None, None, :, :]
                n_samples = len(x_file)
                n_valid_pixels = mask.sum() * n_samples
                total_pixels += n_valid_pixels
                # print(total_pixels)
           
            # Apply mask
            X_masked = x_file * mask_broadcast
            F_masked = f_file * mask_broadcast
            
            # Accumulate sums
            sum_x += np.nansum(X_masked, axis=(0, 2, 3))
            sum_f += np.nansum(F_masked, axis=(0, 2, 3))
            sum_sq_x += np.nansum(X_masked**2 * mask_broadcast, axis=(0, 2, 3))
            sum_sq_f += np.nansum(F_masked**2 * mask_broadcast, axis=(0, 2, 3))
        
        # Calculate mean
        meanx = sum_x / total_pixels
        meanf = sum_f / total_pixels
        
        # Calculate std using accumulated sum of squares
        stdx = np.sqrt(sum_sq_x / total_pixels - meanx**2)
        stdf = np.sqrt(sum_sq_f / total_pixels - meanf**2)
        
        return meanx, stdx, meanf, stdf

    def __getitem__(self, idx):
        """Get item by global index - automatically finds correct file."""
        # Map global index to file and local index
        file_idx, local_idx = self._get_file_and_local_idx(idx)
        
        # Load from the appropriate file
        x = torch.from_numpy(self.X_files[file_idx][local_idx]).float()
        f = torch.from_numpy(self.F_files[file_idx][local_idx]).float()
        
        # Apply transforms
        x = self.tf_x(x)
        f = self.tf_f(f)
        mask = f[[2],:,:]
        
        return x, f, mask

### Masked dataset functions and class. 

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
        
class npyDataResized(MultiFileNpyData):
    def __init__(self, *args, resize_x=(320,320), resize_f=(320,320), **kwargs):
        super().__init__(*args, **kwargs)  # call original __init__

        self.landmask_original = torch.tensor(self.landmask.astype(bool))[None, :, :].float()
        self.landmask_resized = tf.Resize(resize_x)(self.landmask_original)
        self.original_H = self.X_files[0].shape[2]
        self.original_W = self.X_files[0].shape[3]
        
        # Patch the transforms
        self.tf_x = tf.Compose([
            FillNaN(0.0),
            tf.Resize(resize_x),
            tf.Normalize(self.meanx.tolist(), self.stdx.tolist()),
            Mask(self.landmask_resized)
        ])
        self.tf_f = tf.Compose([
            FillNaN(0.0),
            tf.Resize(resize_f),
            tf.Normalize(self.meanf.tolist(), self.stdf.tolist()),
            Mask(self.landmask_resized)
        ])
        # Inverse transforms
        self.inv_tf_x = tf.Compose([
            tf.Resize((self.original_H, self.original_W)),
            tf.Normalize(mean=[0]*len(self.meanx), std=(1/self.stdx).tolist()),
            tf.Normalize(mean=(-self.meanx).tolist(), std=[1]*len(self.stdx)),
            Mask(self.landmask_original)
        ])
        self.inv_tf_f = tf.Compose([
            tf.Resize((self.original_H, self.original_W)),
            tf.Normalize(mean=[0]*len(self.meanf), std=(1/self.stdf).tolist()),
            tf.Normalize(mean=(-self.meanf).tolist(), std=[1]*len(self.stdf)),
            Mask(self.landmask_original)
        ])


# Example usage:
if __name__ == "__main__":
    # Create file list for 12 months
    file_list = [
        (f"data/X_month_{i:02d}.npy", f"data/F_month_{i:02d}.npy")
        for i in range(1, 13)
    ]
    
    # Initialize dataset
    dataset = MultiFileNpyData(
        file_list=file_list,
        maskname="data/mask.npy",
        compute_stats=True
    )
    
    # Random sampling across full year
    from torch.utils.data import DataLoader, RandomSampler
    
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        sampler=sampler,
        num_workers=4
    )
    
    # Iterate through random samples from all months
    for x, f in dataloader:
        print(f"Batch shape: {x.shape}")
        break