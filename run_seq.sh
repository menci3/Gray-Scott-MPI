#!/bin/bash
#SBATCH --job-name=gray-scott-mpi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            
#SBATCH --reservation=fri
#SBATCH --time=03:00:00
#SBATCH --output=out_gray_seq.log

# Load module (if not already loaded)
module load FFmpeg/6.0-GCCcore-12.3.0

# Compile
gcc -O3 -lm sequential.c -o sequential

sizes=("256" "512" "1024" "2048" "4096")

for size in "${sizes[@]}"; do
    echo "Running size=$size sequential"
    srun sequential $size

    ffmpeg -y -framerate 10 -start_number 1 -i frames/frame_%04d.png videos/output_seq_${size}.mp4 -loglevel quiet
done
