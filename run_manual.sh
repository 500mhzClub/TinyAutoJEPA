#!/bin/bash

# 1. Common DDP Environment Variables
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29508   # New port to be safe
export WORLD_SIZE=2

# 2. Critical Safety Flags (Keep these!)
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export RCCL_KERNEL_ENABLE=0   # Force CPU Copy
export HSA_ENABLE_SDMA=0      # Disable hardware copy engines

# 3. Launch Rank 0 (The RDNA4 Card) in the background
# We mask it at the SHELL level so Python never sees the other card.
echo "Starting Rank 0 on GPU 0..."
HIP_VISIBLE_DEVICES=0 \
RANK=0 \
LOCAL_RANK=0 \
python3 train_encoder_parallel.py > rank0.log 2>&1 &
PID_0=$!

# 4. Launch Rank 1 (The RDNA2 Card) in the foreground
echo "Starting Rank 1 on GPU 1..."
HIP_VISIBLE_DEVICES=1 \
RANK=1 \
LOCAL_RANK=1 \
python3 train_encoder_parallel.py 2>&1 | tee rank1.log

# 5. Cleanup: If Rank 1 stops, kill Rank 0
kill $PID_0