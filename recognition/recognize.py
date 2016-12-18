import json

import tensorflow as tf
from scipy.misc import imread

from recognition.train_utils import build_forward
from recognition.utils.train_utils import filter_rectangles


class Recognizer:
    def __init__(self):
        hypes_file = './recognition/data/config/overfeat_rezoom.json'
        with open(hypes_file, 'r') as f:
            config = json.load(f)

        tf.reset_default_graph()
        self.x_in = tf.placeholder(tf.float32, name='x_in', shape=[config['image_height'], config['image_width'], 3])

        if config['use_rezoom']:
            pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(config,
                                                                                                            tf.expand_dims(
                                                                                                                self.x_in,
                                                                                                                0),
                                                                                                            'test',
                                                                                                            reuse=None)
            grid_area = config['grid_height'] * config['grid_width']
            pred_confidences = tf.reshape(
                tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * config['rnn_len'], 2])),
                [grid_area, config['rnn_len'], 2])
            if config['reregress']:
                pred_boxes = pred_boxes + pred_boxes_deltas
        else:
            pred_boxes, pred_logits, pred_confidences = build_forward(config, tf.expand_dims(self.x_in, 0), 'test',
                                                                      reuse=None)

        self.pred_boxes = pred_boxes
        self.pred_confidences = pred_confidences
        self.config = config

        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        saver.restore(self.sess, './recognition/data/save.ckpt-150000v2')

    def recognize(self, image_path):
        img = imread(image_path)
        feed = {self.x_in: img}
        (np_pred_boxes, np_pred_confidences) = self.sess.run([self.pred_boxes, self.pred_confidences], feed_dict=feed)

        filtered = filter_rectangles(self.config, np_pred_confidences, np_pred_boxes, use_stitching=True,
                                     rnn_len=self.config['rnn_len'], min_conf=0.7)
        return filtered


if __name__ == '__main__':
    recognizer = Recognizer()
    recognizer.recognize(image_path='/Users/nickolay/Documents/projects/people-recognition/test_images/t3.jpg')
