model_name=LightTrackM
path_name=back_04502514044521042540+cls_211000022+reg_100000111_ops_32
# test
python tracking/test_lighttrack.py --arch LightTrackM_Subnet --dataset VOT2019 \
--resume snapshot/${model_name}/LightTrackM.pth \
--stride 16 --even 0 --path_name ${path_name}
# evaluation
python lib/eval_toolkit/bin/eval.py --dataset_dir dataset --dataset VOT2019 \
--tracker_result_dir result/VOT2019/ \
--trackers ${model_name}