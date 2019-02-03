set -x
set -e
export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=pylib/src

python test_pixel_link.py \
     --checkpoint_path=$2 \
     --dataset_dir=$3\
     --model_name=$4\
     --gpu_memory_fraction=-1
     

