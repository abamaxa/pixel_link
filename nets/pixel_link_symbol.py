import tensorflow as tf
slim = tf.contrib.slim

from model_variants import *

import config
from nets import basenet
import conv_blocks

class PixelLinkNet(object):
    def __init__(self, inputs, is_training):
        self.inputs = inputs
        self.pixel_cls_logits_add = None
        self.pixel_link_logits_add = None

        self.pixel_cls_logits_flatten = None
        self.pixel_cls_scores_flatten = None
        self.pixel_pos_scores = None
        self.pixel_link_logits = None
        self.link_pos_scores = None
        self.pixel_cls_logits_flatten_add = None
        self.pixel_cls_scores_flatten_add = None
        self.pixel_pos_scores_add = None
        self.pixel_link_logits_add = None
        self.link_pos_scores_add = None

        self.pixel_link_neg_loss_weight_lambda = None
        self.pixel_link_loss_weight = None

        self.pixel_link_labels = None
        self.pixel_link_weights = None

        self.is_training = is_training
        self.pos_mask = None
        self.neg_mask = None
        self.n_pos = None
        self.pixel_cls_loss_weight_lambda = None
        self.pos_pixel_weights_flatten = None

        self._build_network()
        self._fuse_feature_layers()
        self._logits_to_scores()
                
    def _build_network(self):
        with slim.arg_scope([slim.conv2d, slim.separable_convolution2d],
                    activation_fn=tf.nn.relu,
                    weights_regularizer=slim.l2_regularizer(config.weight_decay),
                    weights_initializer= tf.contrib.layers.xavier_initializer(),
                    biases_initializer = tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.separable_convolution2d, 
                                slim.max_pool2d, conv_blocks.expanded_conv],
                                padding='SAME') as sc:
                self.arg_scope = sc
                self.net, self.end_points = basenet.mobilenet_v2_net(
                            inputs=self.inputs, is_training=self.is_training)
        
    def _score_layer(self, input_layer, num_classes, scope):
        with slim.arg_scope(self.arg_scope):
            logits = slim.conv2d(input_layer, num_classes, [1, 1], 
                 stride=1,
                 activation_fn=None, 
                 scope='score_from_%s'%scope,
                 normalizer_fn=None)
            try:
                use_dropout = config.dropout_ratio > 0
            except:
                use_dropout = False
                
            if use_dropout:
                if self.is_training:
                    dropout_ratio = config.dropout_ratio
                else:
                    dropout_ratio = 0
                keep_prob = 1.0 - dropout_ratio
                tf.logging.info('Using Dropout, with keep_prob = %f'%(keep_prob))
                logits = tf.nn.dropout(logits, keep_prob)
            return logits
        
    def _upscore_layer(self, layer, target_layer):   
#             target_shape = target_layer.shape[1:-1] # NHWC
            target_shape = tf.shape(target_layer)[1:-1]
            upscored = tf.image.resize_images(layer, target_shape)
            return upscored        

    def __fuse_layers(self, num_classes, scope, fuse_function) :
        with tf.variable_scope(scope):
            smaller_score_map = None
            for idx, current_layer in enumerate(self.end_points) :
                current_layer_name = "pool{}".format(len(self.end_points) - idx + 1)
                current_score_map = self._score_layer(current_layer, 
                                      num_classes, current_layer_name)
                if smaller_score_map is None:
                    smaller_score_map = current_score_map
                else:
                    smaller_score_map = fuse_function(current_score_map, smaller_score_map)

        return smaller_score_map

    def __fuse_upsample_sum(self, current_score_map, smaller_score_map) :
        upscore_map = self._upscore_layer(smaller_score_map, current_score_map)
        return current_score_map + upscore_map

    def __fuse_concat_sum(self, current_score_map, smaller_score_map) :
        upscore_map = self._upscore_layer(smaller_score_map, current_score_map)
        return tf.concat([current_score_map, upscore_map], axis = 0)

    def __fuse_deconv_sum(self, current_score_map, smaller_score_map) :
        num_outputs = current_score_map.shape[-1].value
        upscore_map = slim.conv2d_transpose(smaller_score_map, 
            num_outputs , [3,3], stride=2)
        upscore_map = slim.conv2d(upscore_map, num_outputs, 1)
        upscore_map = slim.conv2d(upscore_map, num_outputs, 3)
        return current_score_map + upscore_map

    def _fuse_feature_layers(self):
        num_pixel_classes = config.num_classes
        num_link_classes = config.num_neighbours * 2
        num_combined_classes = num_pixel_classes * num_link_classes # 32

        if config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_upsample_sum:
            self.pixel_cls_logits = self.__fuse_layers(num_pixel_classes, 
                scope = 'pixel_cls', fuse_function = self.__fuse_upsample_sum)
            
            self.pixel_link_logits = self.__fuse_layers(num_link_classes, 
                scope = 'pixel_link', fuse_function = self.__fuse_upsample_sum)
            
        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_upsamle_sum_conv1x1_2:
            """
            The feature fuse fashion of 
                'Deep Direct Regression for Multi-Oriented Scene Text Detection'
            
            Instead of fusion of scores, feature map from 1x1, 128 conv are fused,
            and the scores are predicted on it.
            """
            base_map = self.__fuse_layers(num_classes = 128, 
                scope = 'fuse_feature', fuse_function = self.__fuse_upsample_sum)

            self.pixel_cls_logits = self._score_layer(base_map,
                  num_pixel_classes, scope = 'pixel_cls')
            
            self.pixel_link_logits = self._score_layer(base_map,
                   num_link_classes, scope = 'pixel_link')

        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_deconv_sum_conv1x1_2:
            """
            The feature fuse fashion of 
                'Deep Direct Regression for Multi-Oriented Scene Text Detection'
            
            Instead of fusion of scores, feature map from 1x1, 128 conv are fused,
            and the scores are predicted on it.
            """
            base_map = self.__fuse_layers(num_classes = 128, 
                scope = 'fuse_feature', fuse_function = self.__fuse_deconv_sum)

            self.pixel_cls_logits = self._score_layer(base_map,
                  num_pixel_classes, scope = 'pixel_cls')
            
            self.pixel_link_logits = self._score_layer(base_map,
                   num_link_classes, scope = 'pixel_link')

        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_128_upsamle_concat_conv1x1_2:
            base_map = self.__fuse_layers(num_classes = num_combined_classes, 
                scope = 'fuse_feature', fuse_function = self.__fuse_concat_sum)

            self.pixel_cls_logits = self._score_layer(base_map,
                  num_pixel_classes, scope = 'pixel_cls')
            
            self.pixel_link_logits = self._score_layer(base_map,
                   num_link_classes, scope = 'pixel_link')

        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_deconv_sum :
            self.pixel_cls_logits = self.__fuse_layers(num_pixel_classes, 
                scope = 'pixel_cls', fuse_function = self.__fuse_deconv_sum)
            
            self.pixel_link_logits = self.__fuse_layers(num_link_classes, 
                scope = 'pixel_link', fuse_function = self.__fuse_deconv_sum)
        
        elif config.feat_fuse_type == FUSE_TYPE_cascade_conv1x1_deconv_sum_conv1x1_2 :
            #base_map = self._fuse_by_cascade_conv1x1_deconv_sum(32, scope = 'fuse_feature')
            base_map = self.__fuse_layers(num_classes = num_combined_classes, 
                scope = 'fuse_feature', fuse_function = self.__fuse_deconv_sum)
            
            self.pixel_cls_logits = self._score_layer(base_map,
                  num_pixel_classes, scope = 'pixel_cls')
            
            self.pixel_link_logits = self._score_layer(base_map,
                   num_link_classes, scope = 'pixel_link')

            self.pixel_cls_logits_add = self.__fuse_layers(num_pixel_classes, 
                scope = 'pixel_cls_add', fuse_function = self.__fuse_upsample_sum)
            
            self.pixel_link_logits_add = self.__fuse_layers(num_link_classes, 
                scope = 'pixel_link_add', fuse_function = self.__fuse_upsample_sum)

        else:
            raise ValueError('feat_fuse_type not supported:%s'%(config.feat_fuse_type))
        
    def _flat_pixel_cls_values(self, values):
        shape = values.shape.as_list()
        values = tf.reshape(values, shape = [shape[0], -1, shape[-1]])
        return values

    def _logits_to_scores(self):
        self.pixel_cls_logits_flatten, self.pixel_cls_scores_flatten, self.pixel_pos_scores = \
            self.__pixel_cls_logit_to_scores(self.pixel_cls_logits)

        self.pixel_link_logits, self.link_pos_scores = \
            self.__pixel_link_logit_to_scores(self.pixel_link_logits)
            
        if not self.pixel_cls_logits_add is None :
            self.pixel_cls_logits_flatten_add, self.pixel_cls_scores_flatten_add, self.pixel_pos_scores_add = \
                self.__pixel_cls_logit_to_scores(self.pixel_cls_logits_add)

            self.pixel_link_logits_add, self.link_pos_scores_add = \
                self.__pixel_link_logit_to_scores(self.pixel_link_logits_add)

    def __pixel_cls_logit_to_scores(self, pixel_cls_logits) :
        pixel_cls_scores = tf.nn.softmax(pixel_cls_logits)
        pixel_cls_logits_flatten = \
            self._flat_pixel_cls_values(pixel_cls_logits)
        pixel_cls_scores_flatten = \
            self._flat_pixel_cls_values(pixel_cls_scores)
        
        pixel_pos_scores = pixel_cls_scores[:, :, :, 1]
        return pixel_cls_logits_flatten, pixel_cls_scores_flatten, pixel_pos_scores

    def __pixel_link_logit_to_scores(self, pixel_link_logits) :
        shape = tf.shape(pixel_link_logits)
        pixel_link_logits = tf.reshape(pixel_link_logits, 
                                [shape[0], shape[1] * shape[2], config.num_neighbours, 2])
            
        pixel_link_scores = tf.nn.softmax(pixel_link_logits)
        link_pos_scores = pixel_link_scores[:, :, :, 1]
        return pixel_link_logits, link_pos_scores
        
    def build_loss(self, pixel_cls_labels, pixel_cls_weights, 
                        pixel_link_labels, pixel_link_weights,
                        do_summary = True
                        ):      
        """
        The loss consists of two parts: pixel_cls_loss + link_cls_loss, 
            and link_cls_loss is calculated only on positive pixels
        """
        pixel_cls_loss_add = None

        batch_size = config.batch_size_per_gpu
        background_label = config.background_label
        text_label = config.text_label
        self.pixel_link_neg_loss_weight_lambda = config.pixel_link_neg_loss_weight_lambda
        self.pixel_cls_loss_weight_lambda = config.pixel_cls_loss_weight_lambda
        self.pixel_link_loss_weight = config.pixel_link_loss_weight

        self.pixel_link_labels = pixel_link_labels
        self.pixel_link_weights = pixel_link_weights

        # OHNM on pixel classification task
        pixel_cls_labels_flatten = tf.reshape(pixel_cls_labels, [batch_size, -1])
        self.pos_pixel_weights_flatten = tf.reshape(pixel_cls_weights, [batch_size, -1])
        
        self.pos_mask = tf.equal(pixel_cls_labels_flatten, text_label)
        self.neg_mask = tf.equal(pixel_cls_labels_flatten, background_label)

        self.n_pos = tf.reduce_sum(tf.cast(self.pos_mask, dtype = tf.float32))

        pixel_cls_loss = self.__add_pixel_loss('pixel_cls_loss', self.pixel_cls_logits_flatten, self.pixel_cls_scores_flatten)
        if not self.pixel_cls_logits_flatten_add is None :
            pixel_cls_loss_add = self.__add_pixel_loss('pixel_cls_loss_add', 
                self.pixel_cls_logits_flatten_add, 
                self.pixel_cls_scores_flatten_add)

        pixel_pos_link_loss, pixel_neg_link_loss = self.__add_link_loss('pixel_link_loss', self.pixel_link_logits)
        if not self.pixel_link_logits_add is None :
            pixel_pos_link_loss_add, pixel_neg_link_loss_add = self.__add_link_loss('pixel_link_loss_add', self.pixel_link_logits_add)
            
        if do_summary:
            tf.summary.scalar('pixel_cls_loss', pixel_cls_loss)
            tf.summary.scalar('pixel_pos_link_loss', pixel_pos_link_loss)
            tf.summary.scalar('pixel_neg_link_loss', pixel_neg_link_loss)
            if not pixel_cls_loss_add is None :
                tf.summary.scalar('pixel_cls_loss_add', pixel_cls_loss_add)
                tf.summary.scalar('pixel_pos_link_loss_add', pixel_pos_link_loss_add)
                tf.summary.scalar('pixel_neg_link_loss_add', pixel_neg_link_loss_add)

    def __add_pixel_loss(self, scope_name, pixel_cls_logits_flatten, pixel_cls_scores_flatten) :
        with tf.name_scope(scope_name):            
            def has_pos():
                pixel_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = pixel_cls_logits_flatten, 
                    labels = tf.cast(self.pos_mask, dtype = tf.int32))
                
                pixel_neg_scores = pixel_cls_scores_flatten[:, :, 0]
                selected_neg_pixel_mask = OHNM_batch(pixel_neg_scores, self.pos_mask, self.neg_mask)
                
                pixel_cls_weights = self.pos_pixel_weights_flatten + \
                            tf.cast(selected_neg_pixel_mask, tf.float32)
                n_neg = tf.cast(tf.reduce_sum(selected_neg_pixel_mask), tf.float32)
                loss = tf.reduce_sum(pixel_cls_loss * pixel_cls_weights) / (n_neg + self.n_pos)
                return loss
            
#             pixel_cls_loss = tf.cond(n_pos > 0, has_pos, no_pos)
            pixel_cls_loss = has_pos()
            tf.add_to_collection(tf.GraphKeys.LOSSES, pixel_cls_loss * self.pixel_cls_loss_weight_lambda)

        return pixel_cls_loss

    def __add_link_loss(self, scope_name, pixel_link_logits) :
        with tf.name_scope(scope_name):
            def no_pos():
                return tf.constant(.0), tf.constant(.0)
            
            def has_pos():
                pixel_link_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits = pixel_link_logits, 
                    labels = self.pixel_link_labels)
                
                def get_loss(label):
                    link_mask = tf.equal(self.pixel_link_labels, label)
                    link_weights = self.pixel_link_weights * tf.cast(link_mask, tf.float32)
                    n_links = tf.reduce_sum(link_weights)
                    loss = tf.reduce_sum(pixel_link_loss * link_weights) / n_links
                    return loss
                
                neg_loss = get_loss(0)
                pos_loss = get_loss(1)
                return neg_loss, pos_loss
            
            pixel_neg_link_loss, pixel_pos_link_loss = \
                        tf.cond(self.n_pos > 0, has_pos, no_pos)
            
            pixel_link_loss = pixel_pos_link_loss + \
                    pixel_neg_link_loss * self.pixel_link_neg_loss_weight_lambda
                    
            tf.add_to_collection(tf.GraphKeys.LOSSES, 
                                 self.pixel_link_loss_weight * pixel_link_loss)

            return pixel_pos_link_loss, pixel_neg_link_loss
        
def OHNM_single_image(scores, n_pos, neg_mask):
    """Online Hard Negative Mining.
        scores: the scores of being predicted as negative cls
        n_pos: the number of positive samples 
        neg_mask: mask of negative samples
        Return:
            the mask of selected negative samples.
            if n_pos == 0, top 10000 negative samples will be selected.
    """
    def has_pos():
        return n_pos * config.max_neg_pos_ratio
    def no_pos():
        return tf.constant(10000, dtype = tf.int32)
    
    n_neg = tf.cond(n_pos > 0, has_pos, no_pos)
    max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))
        
    n_neg = tf.minimum(n_neg, max_neg_entries)
    n_neg = tf.cast(n_neg, tf.int32)
    def has_neg():
        neg_conf = tf.boolean_mask(scores, neg_mask)
        vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)
        threshold = vals[-1]# a negtive value
        selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
        return selected_neg_mask
    def no_neg():
        selected_neg_mask = tf.zeros_like(neg_mask)
        return selected_neg_mask
    
    selected_neg_mask = tf.cond(n_neg > 0, has_neg, no_neg)
    return tf.cast(selected_neg_mask, tf.int32)

def OHNM_batch(neg_conf, pos_mask, neg_mask):
    selected_neg_mask = []
    for image_idx in xrange(config.batch_size_per_gpu):
        image_neg_conf = neg_conf[image_idx, :]
        image_neg_mask = neg_mask[image_idx, :]
        image_pos_mask = pos_mask[image_idx, :]
        n_pos = tf.reduce_sum(tf.cast(image_pos_mask, tf.int32))
        selected_neg_mask.append(OHNM_single_image(image_neg_conf, n_pos, image_neg_mask))
        
    selected_neg_mask = tf.stack(selected_neg_mask)
    return selected_neg_mask