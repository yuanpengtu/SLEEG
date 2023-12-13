export GPUS=4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python3 tools/anomaly/eval_anomaly.py --config configs/exps/deeplabv3_mst.py --checkpoint results/latest.pth --img-root datasets/leftImg8bit/test/ --gt-root datasets/gtCoarse/test/ --gt-suffix gtCoarse_labelTrainIds.png --img-suffix leftImg8bit.png  
