import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import platform
# need this if you want to inference a trt optimized graph
if platform.system() == "Linux":
    from tensorflow.contrib import tensorrt as trt
import time

def load_test_picture(filename):
    """
    @param filename: location of the image file to load
    @return: image file as numpy array
    """	
    img = cv2.imread(filename, 0)
    print(img.shape)
    img = cv2.resize(img, (28, 28))
    print(img.shape)
    img = img.reshape(1, 1, 28, 28)
    print("reshaping data {}".format(img.shape))
    img = img.astype('float32')
    img /= 255

    return img

def load_graph(frozen_graph_filename):
    """
    @param frozen_graph_filename: location of the .pb file of frozen graph
    @return: tensorflow graph definition
    """	
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")

    return graph

def inference_main(path_frozen_pb, output_node, input_node, img):
    """
    @param path_frozen_pb: location of the .pb file of frozen graph
    @param output_node: name of the ANN's output node
    @parem input_node: name of the input node
    @param img: image to recognize as numpy array
    """		
    if not tf.gfile.Exists(path_frozen_pb):
        raise AssertionError(
                "Model directory doesn't exists. Please specify an import "
                "file: %s" % path_frozen_pb)

    if not output_node:
        print("You need to supply the name of a node to --output_node_names.")
        return -1
    if not input_node:
        print("You need to supply the name of a node to --input_node_name.")
        return -1

    # this is necesary for NVIDIA-Turing Tooling
    # otherwise GPU Memory will be flooded and 
    # cuDNN results in an error
    conf = tf.ConfigProto()
    conf.gpu_options.allow_growth = True

    calib_graph = load_graph(path_frozen_pb)

    y_node = calib_graph.get_tensor_by_name('prefix/' + output_node + ':0')

    sess = tf.Session(config=conf, graph=calib_graph)

    # make the prediction
    start = time.time()
    prediction = sess.run(y_node, feed_dict={'prefix/' + input_node + ':0': img})
    end = time.time()
    prediction = prediction.reshape(10)
    print(prediction)

    print("\n\n###############################")
    print("your number was recognized as {}".format(np.argmax(prediction)))
    print("###############################")
    print("duration for inference: {}s".format(end-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="", help="path to frozen_model.pb")
    parser.add_argument("--output_node_names", type=str, default=None, help="The name of the output nodes, comma separated.")
    parser.add_argument("--input_node_name", type=str, default=None, help="Name of input node")
    parser.add_argument("--picture", type=str, default="", help="path to a picter you want to predict")
    args = parser.parse_args()

    # run the main function
    img = load_test_picture(args.picture)
    inference_main(args.model_path, args.output_node_names, args.input_node_name, img)
