#!/bin/bash
#SBATCH --job-name=gray-scott-mpi
#SBATCH --ntasks=64                  # Total MPI processes
#SBATCH --cpus-per-task=1            # One core per MPI task
#SBATCH --reservation=fri
#SBATCH --time=02:00:00
#SBATCH --output=out_gray_mpi.log

# Load modules (if not already loaded)
module load OpenMPI/4.1.6-GCC-13.2.0
module load FFmpeg/6.0-GCCcore-12.3.0

# Compile
mpicc -O3 -o block_mpi block_mpi.c -lm

sizes=("256" "512" "1024" "2048" "4096")
cores=("1" "2" "4" "16" "32" "64")

for size in "${sizes[@]}"; do
  for core in "${cores[@]}"; do
    echo "Running size=$size cores=$core"
    mpirun -np $core ./block_mpi $size

    ffmpeg -y -framerate 10 -start_number 1 -i frames/frame_%04d.png videos/output_${core}_${size}.mp4 -loglevel quiet
  done
done
