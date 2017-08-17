from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def testOclOp(self):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    conf = tf.ConfigProto(allow_soft_placement=False, device_count={'SYCL': 2})
    sess = tf.InteractiveSession(config=conf)
    with tf.device('/cpu:0'):
        input_tensor = tf.zeros([6], dtype=tf.float32)
        args = tf.fill([1], 42.0)
    with tf.device('/device:SYCL:0'):
        with tf.name_scope('operation'):
            result = tf.user_ops.ocl_native_op(input_list=[input_tensor, args], output_type=tf.float32,
                                               shape=[6],
                                               filename="/workfile/Development/Eigen/opencl/unsupported/Eigen/CXX11/src/Tensor/OpenCL_kernels/Add_arg.cl", kernelname="vector_add")
    print(result.eval())

if __name__ == "__main__":
    tf.app.run(main=testOclOp)
