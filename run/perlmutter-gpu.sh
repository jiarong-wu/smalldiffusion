#!/bin/bash 

#SBATCH -J test1
#SBATCH -q regular 
#SBATCH -C gpu 
#SBATCH -N 1 
#SBATCH -G 4
#SBATCH -t 4:00:00 
#SBATCH -A m4874
#SBATCH --gpu-power=200
#SBATCH -o %x-%j.out

#Other settings go here
module load pytorch/2.6.0
accelerate launch --multi_gpu /scratch/jw8736/smalldiffusion/examples/wavediff_global.py

# srun -n 8 -c 32 --cpu-bind=cores -G 8 --gpu-bind=none ./a.out 
# accelerate launch --multi_gpu ----num_machines=2 --machine_rank=0 wavediff_global.py