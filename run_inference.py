import os
import argparse
import pickle
import numpy as np
from scipy import ndimage
from PIL import Image
import tensorflow as tf
from core.utils import load_coco_data, decode_captions
from core.model import CaptionGenerator

class CaptionInference(object):
    def __init__(self, sess, model_path, use_inception):
        # word to index mapping
        with open('./data/train/word_to_idx.pkl') as f:
            self.word_to_idx = pickle.load(f)

        if use_inception:
            L = 64
            D = 2048
            cnn_model_path = './data/inception_v3.ckpt'
        else:
            L = 196
            D = 512
            cnn_model_path = './data/imagenet-vgg-verydeep-19.mat'

        self.sess = sess
        self.use_inception = use_inception
        print "Creating model..."
        self.model = CaptionGenerator(self.word_to_idx, dim_feature=[L, D], dim_embed=512,
                                 dim_hidden=1800, n_time_step=16, prev2out=True, 
                                 ctx2out=True, alpha_c=5.0, selector=True, dropout=True, 
                                 use_cnn = "inception" if use_inception else "vgg",
                                 cnn_model_path = cnn_model_path)

        print "Loading CNN weights..."
        self.model.cnn.load_weights(sess)
        print "Building sampler..."
        _, _, self.generated_captions = self.model.build_sampler(max_len=20)

        # initialize model and load weights
        print "Loading LSTM weights..."
        # tf.global_variables_initializer().run()
        saver = tf.train.Saver(self.model.sampler_vars)
        saver.restore(sess, model_path)

    def inference_np(self, images):
        all_gen_cap = self.sess.run(self.generated_captions, feed_dict = {self.model.images: images})
        all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
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
        if os.path.splitext(file_name)[1] == "npy":
            return np.load(file_name)
        else:
            img_np = np.array(self.resize_image(Image.open(file_name), self.model.cnn.image_size))
            # convert grey scale image to 3-channel
            if img_np.ndim == 2:
                img_np = np.stack((img_np,)*3, axis = -1)
            return img_np


    def inference_files(self, image_files):
        print "processing {} images...".format(len(image_files))
        image_batch = np.array([self.preprocess_file(x) for x in image_files]).astype(np.float32)
        if self.use_inception:
            image_batch /= 255.0
            image_batch -= 0.5
            image_batch *= 2.0
        return self.inference_np(image_batch)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_inception", action="store_true", help="use inception network image size (299 * 299)")
    parser.add_argument("--model_path", default="model_best/model-best", help="lstm model path")
    parser.add_argument("image", type=str, nargs='+', help="image file paths")
    args = parser.parse_args()
    with tf.Session() as sess:
        cap_infer = CaptionInference(sess, args.model_path, args.use_inception)
        captions = cap_infer.inference_files(args.image)
        for fname, caption in zip(args.image, captions):
            print fname, '\t', caption

