#ifndef TENSORFLOW_USER_OPS_OCL_OP_H_
#define TENSORFLOW_USER_OPS_OCL_OP_H_

#ifndef EIGEN_USE_SYCL
#define EIGEN_USE_SYCL
#endif

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("OpenCLNativeOp")
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

namespace tensorflow {

typedef Eigen::SyclDevice SYCLDevice;

class OpenCLNativeOp : public OpKernel {
public :
  explicit OpenCLNativeOp(OpKernelConstruction* context) : OpKernel(context) {
    std::cout << "Launch !!! " << std::endl;
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
    const void* inputs[num_inputs];
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, out_shape, &output));

    for(int i = 1; i < num_inputs; i++) {
      inputs[i] = context->input(i).flat<int>().data();
    }
    auto dev = context->eigen_sycl_device();
    std::cout << "Computing !!! " << std::endl;
    output->flat<float>().device(dev) = context->input(0).flat<float>().nativeOCL(inputs, num_inputs, kernel_name, file_name);
    std::cout << "Computing END !!! " << std::endl;
  }

 private:
  string kernel_name;
  string file_name;
  TensorShape out_shape;
  bool log_;
};

#ifdef TENSORFLOW_USE_SYCL
 REGISTER_KERNEL_BUILDER(Name("OpenCLNativeOp").Device(DEVICE_SYCL), OpenCLNativeOp);
#endif

} // namespace tensorflow
#endif // TENSORFLOW_USER_OPS_OCL_OP_H_
