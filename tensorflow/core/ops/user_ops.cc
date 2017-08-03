
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

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
O: Output types
Shape: The shape of the output tensor
OpenCLFile: Name of the OpenCL file containing the kernel
KernelName: Name of the OpenCL kernel
)doc");

} // namespace tensorflow
