set -x
set -e

export PYTHONPATH=pylib/src
export CUDA_VISIBLE_DEVICES=$1

python test_pixel_link_on_any_image.py \
            --checkpoint_path=$2 \
            --dataset_dir=$3 \
            --model_name=$4 \
            --eval_image_width=1224\
            --eval_image_height=1632\
            --pixel_conf_threshold=0.5\
            --link_conf_threshold=0.5
