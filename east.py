import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('text_scale', 512, '')

from nets import mobilenet_v2

FLAGS = tf.app.flags.FLAGS

def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])

def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)

def get_mobilenet(images, conv_def, weight_decay=1e-5, is_training=True) :
    use_points = []
    #map_points = [17, 12, 8, 3]
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        logits, end_points = mobilenet_v2.mobilenet_base(images, num_classes=0, conv_defs = conv_def)
        ep_list = [(k,v.shape[-1].value) for k,v in end_points.items() if k.find('/') == -1]
        ep_list.sort(key=lambda x:int(x[0].split('_')[1]))
        
        start_at = 4
        first = 2
        current_outputs = ep_list[start_at][1]
        last_point = None
        last_name = ''
        use_points.append(end_points[ep_list[first][0]])
        
        for ep_name, num_outputs in ep_list[start_at:] :
            if current_outputs != num_outputs :
                current_outputs = num_outputs
                use_points.append(last_point)
                print("Added", last_name)
                last_point = None
   
            last_name = ep_name
            last_point = end_points[ep_name]
              
        if use_points[-1] != last_point :
            use_points.append(last_point)
            print("Added", last_name)
            
        use_points.reverse()      
        #for point in map_points :
        #    use_points.append(end_points["layer_{}".format(point)])

    return logits, use_points    

def make_text_detector(end_points) :
    num_outputs = np.power(2,3 + len(end_points)) #[None, 128, 64, 32]
    h = None
    g = None
    
    for f in end_points :
        print('Shape of f {}'.format(f.shape))
        if h is None:
            h = f
        else:
            con_x = tf.concat([g, f], axis=-1, name="text_detect")
            c1_1 = slim.conv2d(con_x, num_outputs, 1)
            h = slim.conv2d(c1_1, num_outputs, 3)
            
        if f != end_points[-1] :
            g = unpool(h)
        else:
            g = slim.conv2d(h, num_outputs, 3)
            
        num_outputs >> 1
     
    return make_scoring_layers(g)

def make_scoring_layers(g3_layer) :
    # here we use a slightly different way for regression part,
    # we first use a sigmoid to limit the regression range, and also
    # this is do with the angle map
    F_score = slim.conv2d(g3_layer, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
    # 4 channel of axis aligned bbox and 1 channel rotation angle
    geo_map = slim.conv2d(g3_layer, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
    angle_map = (slim.conv2d(g3_layer, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
    F_geometry = tf.concat([geo_map, angle_map], axis=-1)    
    return F_score, F_geometry    

def model(images, weight_decay=1e-5, is_training=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)
            
    conv_def = mobilenet_v2.V2_DEF_v_large
    #logits, end_points = get_mobilenet_v4(images) 
    logits, end_points = get_mobilenet(images, conv_def, weight_decay, is_training)
    #logits, end_points = get_resnet(images, weight_decay, is_training)
    
    with tf.variable_scope('feature_fusion', values=end_points): #[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            F_score, F_geometry = make_text_detector(end_points)
            
    return F_score, F_geometry

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)
    tf.summary.scalar('classification_dice_loss', loss)
    return loss

def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss
