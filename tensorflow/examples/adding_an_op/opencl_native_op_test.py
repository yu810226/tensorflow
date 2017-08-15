from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def testOclOp(self):
    tf.logging.set_verbosity(tf.logging.DEBUG)
    conf = tf.ConfigProto(allow_soft_placement=False, device_count={'SYCL': 2})
    sess = tf.InteractiveSession(config=conf)
    # with self.test_session(config=tf.ConfigProto(log_device_placement=True)):
    with tf.name_scope('fpga_add_test'):
        with tf.device('/cpu:0'):
            input_tensor = tf.zeros([6], dtype=tf.float32)
            # tf.summary.scalar('input_tensor', input_tensor)
            args = tf.fill([1], 42.0)
            # tf.summary.scalar('argument', args)
        with tf.device('/device:SYCL:1'):
            with tf.name_scope('operation'):
                result = tf.user_ops.ocl_native_op(input_list=[input_tensor, args], output_type=tf.float32,
                                                   shape=[6],
                                                   filename="/workfile/Development/Eigen/opencl/unsupported/Eigen/CXX11/src/Tensor/OpenCL_kernels/Add_arg.cl", kernelname="vector_add")
                print(result.eval())
            # tf.summary.scalar('result', result)
    #merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter('/tmp/tensorflow/logs/test', sess.graph)

    # sum, res = sess.run([merged, result])
    #writer.add_summary(sum)
    #write.close()

if __name__ == "__main__":
    #tf.test.main()
    tf.app.run(main=testOclOp)
