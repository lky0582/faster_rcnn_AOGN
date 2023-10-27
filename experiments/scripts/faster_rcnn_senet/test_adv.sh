
#!/bin/bash
# Usage:
# ./experiments/scripts/fast_rcnn.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/fast_rcnn.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3
array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=60000        #original setting=10000
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=280000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/faster_rcnn_end2end_senet_adv_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


time ./tools/test_without_first.py --gpu ${GPU_ID} \
  --def models_change/${PT_DIR}/${NET}/faster_rcnn_end2end/test_senet512.prototxt \
  --net /data/li/output/faster_rcnn_res_adv/faster_rcnn_adv_iter_10000.caffemodel \
  --imdb ${TEST_IMDB} \
  --cfg 'experiments/cfgs/fast_rcnn_adv_128.yml' \
  --num_dets 2000 \
  --det_thresh 0.00001 \
  ${EXTRA_ARGS}
