#!/bin/bash
# Set the environment variables expected by torch.distributed.launch / torchrun
# when running under SLURM. Source this script *within* the srun context.

if [ -n "$SLURM_STEP_NODELIST" ]; then
    target_nodes="$SLURM_STEP_NODELIST"
else
    target_nodes="$SLURM_NODELIST"
fi

export MASTER_ADDR=$(scontrol show hostnames "$target_nodes" | head -n 1)

export RANK=${SLURM_PROCID}
export WORLD_RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}
export WORLD_SIZE=${SLURM_NTASKS}
export MASTER_PORT=${MASTER_PORT:-29500}

echo "Rank $RANK on $(hostname): Master=$MASTER_ADDR Port=$MASTER_PORT"
