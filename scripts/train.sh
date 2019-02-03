set -x
set -e
echo "Usage: GPUs Batch Directory"

TRAIN_DIR=${PWD}/$4
DATASET_DIR=${PWD}/training_data

export PYTHON=python
export PYTHONPATH=pylib/src
export CUDA_VISIBLE_DEVICES=$1
IMG_PER_GPU=$2

# get the number of gpus
OLD_IFS="$IFS" 
IFS="," 
gpus=($CUDA_VISIBLE_DEVICES) 
IFS="$OLD_IFS"
NUM_GPUS=${#gpus[@]}

# batch_size = num_gpus * IMG_PER_GPU
BATCH_SIZE=`expr $NUM_GPUS \* $IMG_PER_GPU`
MODELNAME=tiny

${PYTHON} train_pixel_link.py \
            --train_dir=${TRAIN_DIR} \
            --num_gpus=${NUM_GPUS} \
            --learning_rate=1e-3\
            --gpu_memory_fraction=-1 \
            --train_image_width=512 \
            --train_image_height=512 \
            --batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=synthtext \
            --dataset_split_name=train \
            --max_number_of_steps=100\
            --checkpoint_path=${CKPT_PATH} \
            --using_moving_average=1 \
            --max_saved_checkpoints=300 \
            --model_name=${MODELNAME}

${PYTHON} train_pixel_link.py \
            --train_dir=${TRAIN_DIR} \
            --num_gpus=${NUM_GPUS} \
            --learning_rate=1e-2\
            --gpu_memory_fraction=-1 \
            --train_image_width=512 \
            --train_image_height=512 \
            --batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=synthtext \
            --dataset_split_name=train \
            --checkpoint_path=${CKPT_PATH} \
            --using_moving_average=1 \
            --max_number_of_steps=100000\
            --max_saved_checkpoints=300 \
            --model_name=${MODELNAME}\
            2>&1 | tee -a ${TRAIN_DIR}/log.log                        

${PYTHON} train_pixel_link.py \
            --train_dir=${TRAIN_DIR} \
            --num_gpus=${NUM_GPUS} \
            --learning_rate=1e-2\
            --gpu_memory_fraction=-1 \
            --train_image_width=512 \
            --train_image_height=512 \
            --batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=icdar2015 \
            --dataset_split_name=train \
            --checkpoint_path=${CKPT_PATH} \
            --using_moving_average=1 \
            --max_number_of_steps=170000\
            --max_saved_checkpoints=300 \
            --model_name=${MODELNAME}\
            2>&1 | tee -a ${TRAIN_DIR}/log.log

${PYTHON} train_pixel_link.py \
            --train_dir=${TRAIN_DIR} \
            --num_gpus=${NUM_GPUS} \
            --learning_rate=1e-2\
            --gpu_memory_fraction=-1 \
            --train_image_width=512 \
            --train_image_height=512 \
            --batch_size=${BATCH_SIZE}\
            --dataset_dir=${DATASET_DIR} \
            --dataset_name=icdar2015 \
            --dataset_split_name=train \
            --checkpoint_path=${CKPT_PATH} \
            --using_moving_average=1 \
            --max_saved_checkpoints=300 \
            --max_number_of_steps=1000000 \
            --model_name=${MODELNAME}\
            2>&1 | tee -a ${TRAIN_DIR}/log.log
