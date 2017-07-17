#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

typedef Eigen::SyclDevice SYCLDevice;

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

class OpenCLNativeOp : public OpKernel {
public :
  explicit OpenCLNativeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {


  }

}
