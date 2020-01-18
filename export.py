import time
import math
import os
import argparse
import tensorflow as tf
from keras.models import load_model, model_from_json
import keras.backend as K;

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--model_path', type=str, default='./model.h5')
FLAGS = parser.parse_args()

from model import *
from losses import *

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    K.set_learning_phase(0)

    # load trained model
    json_file = open(os.path.join('/'.join(FLAGS.model_path.split('/')[0:-1]), 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects={'tf': tf, 'RESIZE_FACTOR': RESIZE_FACTOR})
    with K.get_session() as sess:
        print('Loading %s' % FLAGS.model_path)
        model.load_weights(FLAGS.model_path)
        
        checkpoint_path = tf.train.Saver().save(sess, 'models/checkpoint', global_step=0, latest_filename='checkpoint_state')
        print("Checkpoint saved to:", checkpoint_path)

        print("Input layer:", model.layers[0].name)
        print("Output layers:", [out.op.name for out in model.outputs])

        tf.train.write_graph(sess.graph, 'models/', 'model.pb', as_text=True)

if __name__ == '__main__':
    main()
