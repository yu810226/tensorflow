#ifndef EIGEN_USE_SYCL
#define EIGEN_USE_SYCL
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("Openclnativeop")
  .Attr("I: list(type)")
  .Attr("O: type")
  .Attr("Shape: shape")
  .Attr("OpenCLFile: string")
  .Attr("KernelName: string")
  .Input("in: I")
  .Output("out: O")
  .Doc(R"doc(
Operation that executes any given OpenCL kernel with the given args.

I: List of the input types
O: List of the output types
Shape: The shape of the output tensor
OpenCLFile: Name of the OpenCL file containing the kernel
KernelName: Name of the OpenCL kernel
)doc");


#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif

class OpenCLNativeOp : public OpKernel {
public :
  explicit OpenCLNativeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("OpenCLFile", &file_name));
    OP_REQUIRES_OK(context,
                   context->GetAttr("KernelName", &kernel_name));
    OP_REQUIRES_OK(context,
                   context->GetAttr("Shape", &out_shape));
    log_ = StringPiece(type_string()).starts_with("Log");
  }

  void Compute(OpKernelContext* context) override {

    int num_inputs = context->num_inputs();
    int num_outputs = context->num_outputs();
    cl::sycl::context test1;

    const void* inputs[num_inputs-1];
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output));

    for(int i = 0; i < num_inputs-1; i++) {
      inputs[i] = context->input(i+1).flat<float>().data();
    }
    auto dev = context->eigen_sycl_device();
    output->flat<float>().device(dev) = context->input(0).flat<float>().nativeOCL(inputs, num_inputs-1, kernel_name, file_name);
  }

 private:
  string kernel_name;
  string file_name;
  TensorShape out_shape;
  bool log_;
};

#ifdef TENSORFLOW_USE_SYCL
REGISTER_KERNEL_BUILDER(Name("Openclnativeop").Device(DEVICE_SYCL), OpenCLNativeOp);
#endif

} // namespace tensorflow
