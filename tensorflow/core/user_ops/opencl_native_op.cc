#ifndef EIGEN_USE_SYCL
#define EIGEN_USE_SYCL
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("Openclnativeop")
  .Attr("I: list(type)")
  .Attr("O: type")
  .Attr("Shape: shape")
  .Attr("OpenCLFile: string")
  .Attr("KernelName: string")
  .Attr("IsBinary: bool")
  .Input("in: I")
  .Output("out: O")
  .Doc(R"doc(
Operation that executes any given OpenCL kernel with the given args.

I: List of the input types
O: Output type
Shape: The shape of the output tensor
OpenCLFile: Name of the OpenCL file containing the kernel
KernelName: Name of the OpenCL kernel
IsBinary: Specify if the source is a binary file
)doc");


#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif

template <typename T>
class OpenCLNativeOp : public OpKernel {
public :
  explicit OpenCLNativeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("OpenCLFile", &file_name));
    OP_REQUIRES_OK(context,
                   context->GetAttr("KernelName", &kernel_name));
    OP_REQUIRES_OK(context,
                   context->GetAttr("Shape", &out_shape));
    context->GetAttr("IsBinary", &is_binary);
    log_ = StringPiece(type_string()).starts_with("Log");
  }

  void Compute(OpKernelContext* context) override {

    int num_inputs = context->num_inputs();
    int num_outputs = context->num_outputs();

    const void* inputs[num_inputs-1];
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output));

    for(int i = 0; i < num_inputs-1; i++) {
      inputs[i] = context->input(i+1).flat<T>().data();
    }
    auto dev = context->eigen_sycl_device();
    output->flat<T>().device(dev) = context->input(0).flat<T>().nativeOCL(inputs, num_inputs-1, kernel_name, file_name, is_binary);
  }

 private:
  string kernel_name;
  string file_name;
  TensorShape out_shape;
  bool is_binary;
  bool log_;
};

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_SYCL_NATIVE_OP_KERNEL(T) \
  REGISTER_KERNEL_BUILDER(Name("Openclnativeop").Device(DEVICE_SYCL).TypeConstraint<T>("O").TypeConstraint<T>("I"), OpenCLNativeOp<T>);

  TF_CALL_SYCL_NUMBER_TYPES(REGISTER_SYCL_NATIVE_OP_KERNEL)
#endif

} // namespace tensorflow
