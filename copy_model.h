

python python_utils/do_net_surgery.py \
  --out_net_def models/pascal_voc/SENET/faster_rcnn_end2end_adv/train_senet.prototxt \
  --net_surgery_json models/pascal_voc/SENET/faster_rcnn_end2end_adv/init_weights.json \
  --out_net_file output/fast_rcnn_adv/voc_2007_trainval/train_init_senet.caffemodel
