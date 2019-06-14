from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import shutil
import align.detect_face

def main(args):

    #get distance matrix
    data=[]
    image_files=[]
    for image in os.listdir(args.root_path):
        image_files.append(os.path.join(args.root_path,image))
    images = load_and_align_data(image_files, args.image_size, args.margin, args.gpu_memory_fraction)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
          
            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
            nrof_images = len(image_files)
            # print('Images:')
            # for i in range(nrof_images):
            #     print('%1d: %s' % (i, image_files[i]))
            # print('')
            # # Print distance matrix
            # print('Distance matrix')
            # print('    ', end='')
            # for i in range(nrof_images):
            #     print('    %1d     ' % i, end='')
            # print('')
            for i in range(nrof_images):
                data.append([])
                # print('%1d  ' % i, end='')
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
                    data[-1].append(-dist)
                    # print('  %1.4f  ' % dist, end='')
                # print('')
    # print (data)

    # Compute Affinity Propagation(这下面的preference可以调整)
    Similarity = data
    af = AffinityPropagation(affinity='precomputed',preference=-3).fit(Similarity)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    print (labels)
    n_clusters_ = len(cluster_centers_indices)

    # output faces
    category_dir=args.des_path
    if os.path.exists(category_dir):
        shutil.rmtree(category_dir)
    os.mkdir(category_dir)
    for i in range(len(labels)):
        mypath="./{}/{}".format(category_dir,labels[i])
        if not os.path.exists(mypath):
            os.mkdir(mypath)
        shutil.copy(image_files[i],mypath)


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('root_path', type=str, help='root path')
    parser.add_argument('des_path', type=str, help='destination path')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=0)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)
    
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))