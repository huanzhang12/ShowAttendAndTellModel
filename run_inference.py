from __future__ import division
from __future__ import print_function
import os
import sys
import math
import argparse
from six.moves import cPickle as pickle
import numpy as np
from scipy import ndimage
from PIL import Image
import tensorflow as tf
from core.utils import load_coco_data, decode_captions
from core.model import CaptionGenerator

class CaptionInference(object):
    def __init__(self, sess, model_path, use_inception):

        path_prefix = os.path.dirname(os.path.realpath(__file__))
        # word to index mapping
        with open(os.path.join(path_prefix, 'data/train/word_to_idx.pkl'), "rb") as f:
            self.word_to_idx = pickle.load(f)

        if use_inception:
            L = 64
            D = 2048
            cnn_model_path = os.path.join(path_prefix, 'data/inception_v3.ckpt')
        else:
            L = 196
            D = 512
            cnn_model_path = os.path.join(path_prefix, './data/imagenet-vgg-verydeep-19.mat')

        self.batch_size = 128
        self.sess = sess
        self.use_inception = use_inception
        print("Creating model...")
        self.model = CaptionGenerator(self.word_to_idx, dim_feature=[L, D], dim_embed=512,
                                 dim_hidden=1800, n_time_step=16, prev2out=True, 
                                 ctx2out=True, alpha_c=5.0, selector=True, dropout=True, 
                                 use_cnn = "inception" if use_inception else "vgg",
                                 cnn_model_path = cnn_model_path)

        print("Loading CNN weights...")
        self.model.cnn.load_weights(sess)
        print("Building sampler...")
        _, _, self.generated_captions = self.model.build_sampler(max_len=20)

        # initialize model and load weights
        print("Loading LSTM weights...")
        # tf.global_variables_initializer().run()
        saver = tf.train.Saver(self.model.sampler_vars)
        saver.restore(sess, model_path)

    def inference_np(self, images):
        nimgs = images.shape[0]
        print("Running inference on {} images...".format(nimgs))
        nbatches = int(math.ceil(nimgs / self.batch_size))
        all_decoded = []
        for i in range(nbatches):
            start = i * self.batch_size
            end = (i+1) * self.batch_size
            end = nimgs if end >= nimgs else end
            batch_images = images[start:end]
            print("processing {} images ({} to {})".format(batch_images.shape[0], start + 1, end))
            batch_gen_cap = self.sess.run(self.generated_captions, feed_dict = {self.model.images: batch_images})
            batch_decoded = decode_captions(batch_gen_cap, self.model.idx_to_word)
            all_decoded.extend(batch_decoded)
        return all_decoded

    def resize_image(self, image, image_size):
        width, height = image.size
        if width > height:
            left = (width - height) / 2
            right = width - left
            top = 0
            bottom = height
        else:
            top = (height - width) / 2
            bottom = height - top
            left = 0
            right = width
        image = image.crop((left, top, right, bottom))
        image = image.resize([image_size, image_size], Image.ANTIALIAS)
        return image
    
    def preprocess_file(self, file_name):
        if os.path.splitext(file_name)[1] == ".npy":
            return np.squeeze(np.load(file_name))
        else:
            img_np = np.array(self.resize_image(Image.open(file_name), self.model.cnn.image_size)).astype(np.float32)
            # convert grey scale image to 3-channel
            if self.use_inception:
                img_np /= 255.0
                img_np -= 0.5
                img_np *= 2.0
            if img_np.ndim == 2:
                img_np = np.stack((img_np,)*3, axis = -1)
            return img_np


    def inference_files(self, image_files):
        print("processing {} images...".format(len(image_files)))
        image_batch = np.array([self.preprocess_file(x) for x in image_files])
        return self.inference_np(image_batch)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_inception", action="store_true", help="use inception network image size (299 * 299)")
    parser.add_argument("--model_path", default="model_best/model-best", help="lstm model path")
    parser.add_argument("--output", default="", help="save output to a file")
    parser.add_argument("image", type=str, nargs='+', help="image file paths")
    args = parser.parse_args()
    image_files = sorted(args.image)
    if args.output:
        out_file = open(args.output, "w")
    else:
        out_file = sys.stdout
    with tf.Session() as sess:
        cap_infer = CaptionInference(sess, args.model_path, args.use_inception)
        captions = cap_infer.inference_files(image_files)
        for fname, caption in zip(args.image, captions):
            out_file.write("{}\t{}\n".format(fname, caption))
    if args.output:
        out_file.close()
        print("results saved to {}".format(args.output))

