#encoding = utf-8
import sys
import os

# freezing model won't benefit from GPUs, which might already be in use
os.environ["CUDA_VISIBLE_DEVICES"] = ''

# add pylib/src to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pylib', 'src'))

import tensorflow as tf

import util
from test_pixel_link_on_any_image import config_initialization, load_net_for_inference

slim = tf.contrib.slim
import config

tf.app.flags.DEFINE_string('output', None, 'the path to save frozen model to.')

FLAGS = tf.app.flags.FLAGS

FROZEN_MODEL_DIRECTORY = 'frozen'

def freeze(sess, net) :
    output_node_names = [net.link_pos_scores.name.split(':')[0], 
                        net.pixel_pos_scores.name.split(':')[0]]

    if not net.pixel_pos_scores_add is None :
        output_node_names.extend([
            net.link_pos_scores_add.name.split(':')[0], 
            net.pixel_pos_scores_add.name.split(':')[0]
        ])

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    if not os.path.exists(FROZEN_MODEL_DIRECTORY) :
        os.makedirs(FROZEN_MODEL_DIRECTORY)
        
    with open(os.path.join(FROZEN_MODEL_DIRECTORY, FLAGS.output), 'wb') as f:
      f.write(frozen_graph_def.SerializeToString())

def main(_):
    config_initialization()
    net, saver, _ = load_net_for_inference()
    
    with tf.Session() as sess:
        saver.restore(sess, util.tf.get_latest_ckpt(FLAGS.checkpoint_path))
        freeze(sess, net)
    
if __name__ == '__main__':
    tf.app.run()
    
