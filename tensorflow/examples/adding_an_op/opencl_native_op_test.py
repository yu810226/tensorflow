from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def testOclOp(self):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    conf = tf.ConfigProto(allow_soft_placement=False, device_count={'SYCL': 3})
    sess = tf.InteractiveSession(config=conf)
    with tf.device('/cpu:0'):
        arg1 = tf.fill([6,2,3], -2.0)
        arg2 = tf.fill([6,2,3], 22.0)
        arg3 = tf.fill([6,2,3], 2.0)
    with tf.device('/device:SYCL:2'):
        result = tf.user_ops.ocl_native_op(input_list=[arg1, arg2, arg3], output_type=tf.float32, shape=[6,2,3],
                                           file_name="/workfile/Development/Eigen/opencl/unsupported/Eigen/CXX11/src/Tensor/OpenCL_kernels/Add_arg.cl", kernel_name="vector_add", is_binary=False)
    print(result.eval())

if __name__ == "__main__":
    tf.app.run(main=testOclOp)
