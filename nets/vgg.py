import tensorflow as tf
import mobilenet_v2
import config

slim = tf.contrib.slim

def basenet(inputs, fatness = 64, dilation = True, is_training = False):
    #return __basenet_vgg16(inputs, fatness, dilation)
    return __basenet_mobilenet_v2(inputs, fatness, dilation, is_training)

def __basenet_mobilenet_v2(inputs, fatness = 64, dilation = True, is_training = False):
    
    conv_def = getattr(mobilenet_v2, "V2_DEF_" + config.model_name) #v5
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        logits, end_points = mobilenet_v2.mobilenet_base(inputs, num_classes=0, 
            conv_defs = conv_def, is_training = is_training)

        if conv_def == mobilenet_v2.V2_DEF_v4 :
            layers = [4,8,12,16,17]
            
        elif conv_def == mobilenet_v2.V2_DEF_v5 :
            layers = [6,10,14,17,19]
            
        elif conv_def == mobilenet_v2.V2_DEF_v5s :
            layers = [4,6,8,9,10]
            
        elif conv_def == mobilenet_v2.V2_DEF_pixel :
            layers = [6,10,14,16]

        pool_no = 2
        end_point_map = {}
        for layer_no in layers :
            # or expansion_output
            end_point_map['pool{}'.format(pool_no)] = end_points["layer_{}".format(layer_no)]
            pool_no += 1
         
    return logits, end_point_map    

def __basenet_vgg16(inputs, fatness = 64, dilation = True):
    """
    backbone net of vgg16
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    # Original VGG-16 blocks.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        # Block1
        net = slim.repeat(inputs, 2, slim.conv2d, fatness, [3, 3], scope='conv1')
        end_points['conv1_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        end_points['pool1'] = net
        
        
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, fatness * 2, [3, 3], scope='conv2')
        end_points['conv2_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        end_points['pool2'] = net
        
        
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 4, [3, 3], scope='conv3')
        end_points['conv3_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        end_points['pool3'] = net
        
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv4')
        end_points['conv4_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        end_points['pool4'] = net
        
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, fatness * 8, [3, 3], scope='conv5')
        end_points['conv5_3'] = net
        net = slim.max_pool2d(net, [3, 3], 1, scope='pool5')
        end_points['pool5'] = net

        # fc6 as conv, dilation is added
        if dilation:
            net = slim.conv2d(net, fatness * 16, [3, 3], rate=6, scope='fc6')
        else:
            net = slim.conv2d(net, fatness * 16, [3, 3], scope='fc6')
        end_points['fc6'] = net

        # fc7 as conv
        net = slim.conv2d(net, fatness * 16, [1, 1], scope='fc7')
        end_points['fc7'] = net

    return net, end_points    

