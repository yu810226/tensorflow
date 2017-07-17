#ifndef TENSORFLOW_USER_OPS_OCL_OP_H_
#define TENSORFLOW_USER_OPS_OCL_OP_H_

#define EIGEN_USE_THREADS


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

REGISTER_OP("OpenCLNativeOp")
.Attr("I: list(type)")
.Attr("O: list(type)")
.Attr("OpenCLFile: string")
.Attr("KernelName: string")
.Input("in: I")
.Output("out: O")
.Doc(R"doc(
Operation that executes any given OpenCL kernel with the given args.

I: List of the input types
O: List of the output types
OpenCLFILE: Name of the OpenCL file containing the kernel
KernelName: Name of the OpenCL kernel
)doc");

namespace tensorflow {

typedef Eigen::SyclDevice SYCLDevice;

class OpenCLNativeOp : public OpKernel {
public :
  explicit OpenCLNativeOp(OpKernelConstruction* context) : OpKernel(context) {
    log_ = StringPiece(type_string()).starts_with("Log");
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES_OK(context,
                   context->GetAttr("OpenCLFile", &file_name));
    OP_REQUIRES_OK(context,
                   context->GetAttr("KernelName", &kernel_name));
    int num_inputs = context->num_inputs();
    int num_outputs = context->num_outputs();
    void* inputs[num_inputs];
    Tensor* outputs;
    for(int i = 0; i < num_inputs; i++) {
      inputs[i] = &(context->input(i).tensor().data());
    }
    /*TODO allocate outputs*/
    auto dev = context->eigen_sycl_device();
    outputs.device(dev) = context->input(0).tensor().nativeOCL(kernel_name, file_name, inputs);
  }

 private:
  string kernel_name;
  string file_name;

};

} // namespace tensorflow
#endif // TENSORFLOW_USER_OPS_OCL_OP_H_
