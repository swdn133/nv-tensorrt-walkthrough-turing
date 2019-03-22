import numpy as nppy
import argparse
import tensorflow as tf

from tensorflow.contrib import tensorrt as trt
from tensorflow.python.platform import gfile

def create_optimized_trt_graph(frozen_path, saving_path, output_node, precision):
    """
    @param frozen_path: location of the original unoptimized frozen_graph.pb
    @param saving_path: where do you want the new .pb to be saved?
    @param output_node: name of the ANN's output node
    @precision: precision for optimization (e.g. FP16)
    """ 
    if len(output_node.split(':0')) < 2:
        output_node = output_node + ':0'

    alloc_space_TensorRT = 2
    ppgmf = (8 - alloc_space_TensorRT)/8
    max_workspace_size_bytes = alloc_space_TensorRT*1000000000


    with gfile.FastGFile(frozen_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        trt_graph = trt.create_inference_graph(input_graph_def=graph_def,outputs=[output_node], 
                                               max_batch_size=32,
                                               max_workspace_size_bytes=max_workspace_size_bytes,
                                               minimum_segment_size=1,
                                               precision_mode=precision)
        
        path_new_frozen_pb = saving_path + "/newFrozenModel_TRT_" + precision + ".pb"
        with gfile.FastGFile(path_new_frozen_pb, 'wb') as fp:
            fp.write(trt_graph.SerializeToString())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_path", type=str, default=None, help="path to frozen_model.pb")
    parser.add_argument("--output_node_names", type=str, default=None, help="The name of the output nodes, comma separated.")
    parser.add_argument("--saving_path", type=str, default=None, help="the new .pb model will be saved here")
    parser.add_argument("--precision", type=str, default=None, help="e.g. FP16, FP32")
    args = parser.parse_args()

    # run the main function
    create_optimized_trt_graph(args.frozen_path, args.saving_path, args.output_node_names, args.precision)
