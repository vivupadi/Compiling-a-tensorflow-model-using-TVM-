
# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

# tvm, relay
import tvm
from tvm import te
from tvm import relay
import vta
# os and numpy
import pandas as pd
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
from sklearn.model_selection import train_test_split

file = '/home/daniel/Desktop/Vivek/Corrected Final dataset.csv'
df = pd.read_csv(file)
df = df.iloc[:, 2:]

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_sample = X_test.iloc[200:201,:]
X_sample = np.float32(np.array(X_sample))


# Ask tensorflow to limit its GPU memory to what's actually needed
# instead of gobbling everything that's available.
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
# This way this tutorial is a little more friendly to sphinx-gallery.
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("tensorflow will use experimental.set_memory_growth(True)")
    except RuntimeError as e:
        print("experimental.set_memory_growth option is not available: {}".format(e))


try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

######################################################################
# Target settings
# Use these commented settings to build for cuda.
# target = tvm.target.Target("cuda", host="llvm")
# layout = "NCHW"
# dev = tvm.cuda(0)
target = tvm.target.Target("llvm", host="llvm")
layout = None
dev = tvm.cpu(0)

######################################################################
# Download required files
# -----------------------
# Download files listed above.
from tvm.contrib.download import download_testdata

#img_path = download_testdata(image_url, img_name, module="data")
model_path = '/home/daniel/Desktop/Vivek/ANN_Model/frozen graph/frozen_graph.pb'
#map_proto_path = download_testdata(map_proto_url, map_proto, module="data")
#label_path = download_testdata(label_map_url, label_map, module="data")

######################################################################
# Import model
# ------------
# Creates tensorflow graph definition from protobuf file.
with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
    graph_def = tf_compat_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")
    # Call the utility to import the graph definition into default graph.
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf_compat_v1.Session() as sess:
        graph_def = tf_testing.AddShapesToGraphDef(sess, "Identity")


# Import the graph to Relay
# -------------------------
# Import tensorflow graph definition to relay frontend.
#
# Results:
#   sym: relay expr for given tensorflow protobuf.
#   params: params converted from tensorflow params (tensor protobuf).
shape_dict = {"input_1": X_sample.shape}
#dtype_dict = {"DecodeJpeg/contents": "uint8"}
mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout)#, shape=shape_dict)

print('Eureka...Tensorflow protobuf imported to relay frontend')

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification and evaluate
#
# Results:
#   graph: Final graph after compilation.
#   params: final params after compilation.
#   lib: target library which can be deployed on target with TVM runtime.

#with tvm.transform.PassContext(opt_level=1):
#    lib = relay.build(mod, target, "VM", params=params)
    
ex = tvm.relay.create_executor("vm", mod, tvm.cpu(0), target, params)

result = np.where(ex.evaluate()(X_sample).asnumpy() > 0.5, 1, 0)
print(result)

print('Splendid...Inference is done!!!!')
