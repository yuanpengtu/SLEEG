PATHSTATIC=./data/segmentme/fs_static/
CONFIGMST=./configs/exps/deeplabv3_mst.py
CHECKPOINT=./results/latest.pth
CHECKPOINT1=./results/iter_36000.pth
CHECKPOINT2=./results/iter_32000.pth
CHECKPOINT3=./results/iter_28000.pth
CHECKPOINT4=./results/iter_24000.pth

python3 tools/anomaly/eval_anomaly.py --img-root $PATHSTATIC --gt-root $PATHSTATIC --img-suffix _rgb.jpg --gt-suffix _labels.png --config $CONFIGMST --checkpoint $CHECKPOINT #&>>$LOG

python3 tools/anomaly/eval_anomaly.py --img-root $PATHSTATIC --gt-root $PATHSTATIC --img-suffix _rgb.jpg --gt-suffix _labels.png --config $CONFIGMST --checkpoint $CHECKPOINT1 #&>>$LOG

python3 tools/anomaly/eval_anomaly.py --img-root $PATHSTATIC --gt-root $PATHSTATIC --img-suffix _rgb.jpg --gt-suffix _labels.png --config $CONFIGMST --checkpoint $CHECKPOINT2 #&>>$LOG

python3 tools/anomaly/eval_anomaly.py --img-root $PATHSTATIC --gt-root $PATHSTATIC --img-suffix _rgb.jpg --gt-suffix _labels.png --config $CONFIGMST --checkpoint $CHECKPOINT3 #&>>$LOG

python3 tools/anomaly/eval_anomaly.py --img-root $PATHSTATIC --gt-root $PATHSTATIC --img-suffix _rgb.jpg --gt-suffix _labels.png --config $CONFIGMST --checkpoint $CHECKPOINT4 #&>>$LOG



