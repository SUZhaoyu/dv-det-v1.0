/* Furthest point sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 * Happy Mid-Autumn Festival! :)
 */
#include <stdio.h>

__global__ void get_gt_bbox_gpu_kernel(int batch_size, int npoint, int nbbox, int bbox_attr,
                                       int diff_thres, int cls_thres, float padding_offset, bool ignore_height,
                                       const float* input_coors,
                                       const float* gt_bbox,
                                       const int* input_num_list,
                                       int* input_accu_list,
                                       float* output_bbox,
                                       int* output_conf) {
    if (batch_size * nbbox * bbox_attr <=0 || npoint <=0) {
//        printf("Get output Logits Op exited unexpectedly.\n");
        return;
    }
//    const float PI = 3.1415927;
//    if (threadIdx.x == 0 && blockIdx.x == 0) {
    input_accu_list[0] = 0;
    for (int b=1; b<batch_size; b++) {
        input_accu_list[b] = input_accu_list[b-1] + input_num_list[b-1];
    }
//    }
    __syncthreads();
//    printf("%d\n", input_accu_list[5]);
    for (int b=blockIdx.x; b<batch_size; b+=gridDim.x) {
        for (int i=threadIdx.x; i<input_num_list[b]; i+=blockDim.x) {

            float point_x = input_coors[input_accu_list[b]*3 + i*3 + 0];
            float point_y = input_coors[input_accu_list[b]*3 + i*3 + 1];
            float point_z = input_coors[input_accu_list[b]*3 + i*3 + 2];

            for (int j=0; j<nbbox; j++) {
            // [w, l, h, x, y, z, r, cls, diff_idx]
            //  0  1  2  3  4  5  6   7      8
                float bbox_w = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 0];
                float bbox_l = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 1];
                float bbox_h = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 2];
                float bbox_x = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 3];
                float bbox_y = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 4];
                float bbox_z = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 5];
                float bbox_r = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 6];
                float bbox_cls = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 7];
                float bbox_diff = gt_bbox[b*nbbox*bbox_attr + j*bbox_attr + 8];
                if (bbox_l*bbox_h*bbox_w > 0) {
                    float rel_point_x = point_x - bbox_x;
                    float rel_point_y = point_y - bbox_y;
                    float rel_point_z = point_z - bbox_z;
                    float rot_rel_point_x = rel_point_x*cosf(bbox_r) + rel_point_y*sinf(bbox_r);
                    float rot_rel_point_y = -rel_point_x*sinf(bbox_r) + rel_point_y*cosf(bbox_r);
                    if (abs(rot_rel_point_x) <= bbox_w / 2 + padding_offset &&
                        abs(rot_rel_point_y) <= bbox_l / 2 + padding_offset &&
                        (abs(rel_point_z) <= bbox_h / 2 + padding_offset || ignore_height)) {

//                        printf("%d\n", b);

                        output_bbox[input_accu_list[b]*7 + i*7 + 0] = bbox_w;
                        output_bbox[input_accu_list[b]*7 + i*7 + 1] = bbox_l;
                        output_bbox[input_accu_list[b]*7 + i*7 + 2] = bbox_h;
                        output_bbox[input_accu_list[b]*7 + i*7 + 3] = bbox_x;
                        output_bbox[input_accu_list[b]*7 + i*7 + 4] = bbox_y;
                        output_bbox[input_accu_list[b]*7 + i*7 + 5] = bbox_z;
                        output_bbox[input_accu_list[b]*7 + i*7 + 6] = bbox_r;

//                        if (bbox_diff <= diff_thres && bbox_cls == 0) {
                        if (bbox_diff <= diff_thres && bbox_cls <= cls_thres) {
//                            printf("%d\n", b);
                            // Here we only take cars into consideration, while vans are excluded and give the foreground labels as -1 (ignored).
                            // TODO: need to change the category class accordingly to the expected detection target.
                            output_conf[input_accu_list[b] + i] = 1;
                        }else{
                            output_conf[input_accu_list[b] + i] = -1;
                        }
                    }
                }
            }
        }
    }
}

void get_gt_bbox_gpu_launcher(int batch_size, int npoint, int nbbox, int bbox_attr,
                              int diff_thres, int cls_thres, float padding_offset, bool ignore_height,
                              const float* input_coors,
                              const float* gt_bbox,
                              const int* input_num_list,
                              int* input_accu_list,
                              float* output_bbox,
                              int* output_conf) {
    get_gt_bbox_gpu_kernel<<<32,512>>>(batch_size, npoint, nbbox, bbox_attr,
                                       diff_thres, cls_thres, padding_offset, ignore_height,
                                       input_coors,
                                       gt_bbox,
                                       input_num_list,
                                       input_accu_list,
                                       output_bbox,
                                       output_conf);
}
