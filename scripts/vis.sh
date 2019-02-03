# !/bin/bash
export PYTHONPATH=pylib/src

python visualize_detection_result.py \
	--image=$1 \
	--det=$2 \
	--output=~/temp/no-use/pixel_result
	
