import argparse
import tensorflow as tf


def freeze_graph(model_dir, output_node_names):
    """
    @param model_dir: location where saved model (chekpoint) is stored
    @param param2: name of the output layer
    @return: output graph definition
    @raise AssertionError: when specified path does not exist
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1


    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True


    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,                                       # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),      # The graph_def is used to retrieve the nodes 
                output_node_names.split(",")                # The output node names are used to select the usefull nodes
                )        


        #Print list of output graph nodes       
        i = 0        
        for x in output_graph_def.node:
            if i < 110:
                print(x.name)
                i = i + 1

        # Finally we serialize and dump the output graph to the filesystem 
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    print("\n\nfrozen graph saved to {}".format(output_graph))
    return output_graph_def

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export/import")
    parser.add_argument("--output_node_names", type=str, default=None, help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    # run the main function
    freeze_graph(args.model_dir, args.output_node_names)
