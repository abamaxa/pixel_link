import tensorflow as tf
import mobilenet_v2
import config

from nets import conv_blocks as ops
from nets import mobilenet as lib

slim = tf.contrib.slim
op = lib.op

expand_input = ops.expand_input_by_factor

MODEL_DEFAULTS = {
    # Note: these parameters of batch norm affect the architecture
    # that's why they are here and not in training_scope.
    (slim.batch_norm,): {'center': True, 'scale': True},
    (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
        'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
    },
    (ops.expanded_conv,): {
        'expansion_size': expand_input(6),
        'split_expansion': 1,
        'normalizer_fn': slim.batch_norm,
        'residual': True
    },
    (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
}

V2_DEF_tiny = dict(
    defaults=MODEL_DEFAULTS,
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=2, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=2, num_outputs=256)
    ],
)

# v4
V2_DEF_small = dict(
    defaults=MODEL_DEFAULTS,
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=2, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=2, num_outputs=256),
        op(ops.expanded_conv, stride=1, num_outputs=256)
    ],
)

V2_DEF_medium = dict(
    defaults=MODEL_DEFAULTS,
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=2, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=2, num_outputs=256),
        op(ops.expanded_conv, stride=1, num_outputs=256),
        op(ops.expanded_conv, stride=2, num_outputs=512)
    ],
)

V2_DEF_large = dict(
    defaults=MODEL_DEFAULTS,
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=2, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=2, num_outputs=256),
        op(ops.expanded_conv, stride=1, num_outputs=256),
        op(ops.expanded_conv, stride=1, num_outputs=256),
        op(ops.expanded_conv, stride=2, num_outputs=512),
        op(ops.expanded_conv, stride=1, num_outputs=512)
    ],
)

V2_DEF_very_large = dict(
    defaults=MODEL_DEFAULTS,
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=2, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=1, num_outputs=128),
        op(ops.expanded_conv, stride=2, num_outputs=256),
        op(ops.expanded_conv, stride=1, num_outputs=256),
        op(ops.expanded_conv, stride=2, num_outputs=512),
        op(ops.expanded_conv, stride=1, num_outputs=512),
        op(ops.expanded_conv, stride=2, num_outputs=1024)
    ],
)

def mobilenet_v2_net(inputs, is_training = False):
    conv_def = globals()["V2_DEF_" + config.model_name] 
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        logits, end_points = mobilenet_v2.mobilenet_base(inputs, num_classes=0, 
            conv_defs = conv_def, is_training = is_training)

        if conv_def == V2_DEF_tiny :
            layers = [5,8,11,12]
            
        elif conv_def == V2_DEF_small :
            layers = [6,10,14,16]        
            
        elif conv_def == V2_DEF_medium :
            layers = [5,8,11,13,14]       

        elif conv_def == V2_DEF_large :
            layers = [6,10,14,17,19]        

        elif conv_def == V2_DEF_very_large :
            layers = [5,8,11,13,15,16]

        if config.strides[0] == 2:
            layers.insert(0, 2)

        pool_no = 2
        end_point_map = []
        for layer_no in layers :
            end_point_map.append(end_points["layer_{}".format(layer_no)])
            pool_no += 1
         
    end_point_map.reverse()
    return logits, end_point_map    
