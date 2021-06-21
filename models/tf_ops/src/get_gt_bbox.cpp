/* get ground truth tensorflow cpu wrapper
 * By Zhaoyu SU, Email: zsuad@connect.ust.hk
 * All Rights Reserved. Sep, 2019.
 */
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::InferenceContext;
using ::tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("GetGtBboxOp")
    .Input("input_coors: float32")
    .Input("gt_bbox: float32")
    .Input("input_num_list: int32")
    .Output("output_bbox: float32")
    .Output("output_conf: int32")
    .Attr("padding_offset: float")
    .Attr("diff_thres: int")
    .Attr("cls_thres: int")
    .Attr("ignore_height: bool")
    .SetShapeFn([](InferenceContext* c){
        ShapeHandle input_coors_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_coors_shape));

        DimensionHandle npoint = c->Dim(input_coors_shape, 0);
        ShapeHandle output_bbox_shape = c->MakeShape({npoint, 7});
        ShapeHandle output_conf_shape = c->MakeShape({npoint});
        c->set_output(0, output_bbox_shape);
        c->set_output(1, output_conf_shape);

        return Status::OK();

    }); // InferenceContext

void get_gt_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr,
                              int diff_thres, int cls_thres, float padding_offset, bool ignore_height,
                              const float* input_coors,
                              const float* gt_bbox,
                              const int* input_num_list,
                              int* input_accu_list,
                              float* output_bbox,
                              int* output_conf);

class GetGtBboxOp: public OpKernel {
public:
    explicit GetGtBboxOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("diff_thres", &diff_thres));
        OP_REQUIRES_OK(context, context->GetAttr("cls_thres", &cls_thres));
        OP_REQUIRES_OK(context, context->GetAttr("padding_offset", &padding_offset));
    }
    void Compute(OpKernelContext* context) override {
        const Tensor& input_coors = context->input(0);
        auto input_coors_ptr = input_coors.template flat<float>().data();
        OP_REQUIRES(context, input_coors.dim_size(1) == 3,
            errors::InvalidArgument("The attribute of lidar coors has to be 3."));

        const Tensor& gt_bbox = context->input(1);
        auto gt_bbox_ptr = gt_bbox.template flat<float>().data();
        OP_REQUIRES(context, gt_bbox.dim_size(2)==9,
                    errors::InvalidArgument("Attribute of bbox has to be 9: [l, h, w, x, y, z, r, cls, diff_idx]."));

        const Tensor& input_num_list = context->input(2);
        auto input_num_list_ptr = input_num_list.template flat<int>().data();
        OP_REQUIRES(context, input_num_list.dims()==1,
                    errors::InvalidArgument("FPS Op expects input in shape: [batch_size]."));

        int batch_size = input_num_list.dim_size(0);
        int bbox_attr = gt_bbox.dim_size(2);
        int npoint = input_coors.dim_size(0);
        int nbbox = gt_bbox.dim_size(1);

        int batch_byte_size = batch_size * sizeof(int);
        int* input_num_list_ptr_host = (int*)malloc(batch_byte_size);
        int* input_accu_list_ptr_host = (int*)malloc(batch_byte_size);
        cudaMemcpy(input_num_list_ptr_host, input_num_list_ptr, batch_byte_size, cudaMemcpyDeviceToHost);
        input_accu_list_ptr_host[0] = 0;

        for (int b=1; b<batch_size; b++) {
            input_accu_list_ptr_host[b] = input_accu_list_ptr_host[b-1] + input_num_list_ptr_host[b-1];
        }

        Tensor input_accu_list;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int>::value,
                                                       TensorShape{batch_size},
                                                       &input_accu_list));
        int* input_accu_list_ptr = input_accu_list.template flat<int>().data();
        cudaMemcpy(input_accu_list_ptr, input_accu_list_ptr_host, batch_byte_size, cudaMemcpyHostToDevice);

        Tensor* output_bbox = nullptr;
        auto output_bbox_shape = TensorShape({npoint, 7});
        OP_REQUIRES_OK(context, context->allocate_output(0, output_bbox_shape, &output_bbox));
        float* output_bbox_ptr = output_bbox->template flat<float>().data();
        cudaMemset(output_bbox_ptr, 0.f, npoint * 7 * sizeof(float));

        Tensor* output_conf = nullptr;
        auto output_conf_shape = TensorShape({npoint});
        OP_REQUIRES_OK(context, context->allocate_output(1, output_conf_shape, &output_conf));
        int* output_conf_ptr = output_conf->template flat<int>().data();
        cudaMemset(output_conf_ptr, 0, npoint * sizeof(int));

        get_gt_bbox_gpu_launcher(batch_size, npoint, nbbox, bbox_attr,
                                 diff_thres, cls_thres, padding_offset, ignore_height,
                                 input_coors_ptr,
                                 gt_bbox_ptr,
                                 input_num_list_ptr,
                                 input_accu_list_ptr,
                                 output_bbox_ptr,
                                 output_conf_ptr);

    }
private:
    int diff_thres, cls_thres;
    float padding_offset;
    bool ignore_height;
};
REGISTER_KERNEL_BUILDER(Name("GetGtBboxOp").Device(DEVICE_GPU), GetGtBboxOp);