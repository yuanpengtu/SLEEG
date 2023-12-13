export NCCL_IB_DISABLE=1
export PORT=8888

CONFIG=./configs/exps/deeplabv3_patch.py

python3 -m torch.distributed.launch \
  --nnodes=$HOST_NUM \
  --node_rank=$INDEX \
  --nproc_per_node $HOST_GPU_NUM \
  --master_addr $CHIEF_IP \
  --master_port $PORT \
  tools/train.py $CONFIG --launcher pytorch
