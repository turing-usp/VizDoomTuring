#!/bin/bash
#SBATCH --job-name=vizdoom_rafa
#SBATCH --output=jobs/job%j_%x.out
#SBATCH --error=jobs/job%j_%x.err
#SBATCH --nodes=1
#SBATCH --partition=GPUs
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=rafaelromanrezende123@usp.br    # <-- COLOQUE O SEU EMAIL AQUI
#SBATCH --cpus-per-task=12

echo "========================================================"
echo "Job SLURM iniciado em: $(date)"
echo "Nó alocado: $SLURMD_NODENAME"
echo "========================================================"

# O comando do Singularity com aceleração de GPU
singularity exec --nv --bind .:/app --pwd /app vizdoom_env.sif python -m vizdm_comp.framework.distributed_train_multi \
  --num-matches 20 \
  --game-port 6000 \
  --game-ip 127.0.0.1 \
  --timelimit 999 \
  --stack 4 \
  --trainer-host 127.0.0.1 \
  --trainer-port 7799 \
  --auth-key vizdoom_dm \
  --chunk-steps 5000000 \
  --agent vizdm_comp/framework/tag_pegador.yaml:1 \
  --agent vizdm_comp/framework/tag_fugitivo.yaml:1 \
  --map map01 \
  --wad freedm.wad \
  --shm-obs \
  --game-config vizdm_comp/framework/tag.cfg