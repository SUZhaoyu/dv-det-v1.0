/* Voxel sampling GPU implementation
 * Author Zhaoyu SU
 * All Rights Reserved. Sep., 2019.
 */
#include <stdio.h>
#include <vector>

__device__ inline int get_batch_id(int* accu_list, int batch_size, int id) {
    for (int b=0; b<batch_size-1; b++) {
        if (id >= accu_list[b]) {
            if(id < accu_list[b+1])
                return b;
        }
    }
    return batch_size - 1;
}

__global__ void grid_sampling_gpu_kernel(int batch_size, int input_point_num,
                                         float res_w, float res_l, float res_h,
                                         int grid_w, int grid_l, int grid_h,
                                         const float* input_coors,
                                         const int* input_num_list,
                                         int* input_accu_list,
                                         int* output_idx_temp,
                                         int* output_num_list,
                                         int* grid_buffer) {
    if (batch_size*input_point_num <=0) {
        printf("GridSamplingOp exits due to void inputs.\n");
        return;
    }
	int grid_size = grid_w * grid_l * grid_h;

	int point_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (point_id < input_point_num) {
        int batch_id = get_batch_id(input_accu_list, batch_size, point_id);
        int grid_coor_w = __float2int_rz(input_coors[point_id*3 + 0] / res_w);
        int grid_coor_l = __float2int_rz(input_coors[point_id*3 + 1] / res_l);
        int grid_coor_h = __float2int_rz(input_coors[point_id*3 + 2] / res_h);
//        printf("[%d, %d, %d]\n", grid_coor_w, grid_coor_l, grid_coor_h);
        int grid_buffer_idx = batch_id * grid_size + grid_coor_w * grid_l * grid_h + grid_coor_l * grid_h + grid_coor_h;
        int ret = atomicAdd(&grid_buffer[grid_buffer_idx], 1);

        if (ret == 0) {
            int count = atomicAdd(&output_num_list[batch_id], 1);
            output_idx_temp[input_accu_list[batch_id] + count] = point_id;
        }
    }
}


void grid_sampling_gpu_launcher(int batch_size, int input_point_num,
                                std::vector<float> resolution,
                                std::vector<int> grid_dims,
                                const float* input_coors,
                                const int* input_num_list,
                                int* input_accu_list,
                                int* output_idx_temp,
                                int* output_num_list,
                                int* grid_buffer) {
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch
    int gridSize;       // The actual grid size needed, based on input size
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, grid_sampling_gpu_kernel, 0, input_point_num);
    // Round up according to array size
    gridSize = (input_point_num + blockSize - 1) / blockSize;
//    printf("***************Here******************\n");
//    printf("batch_size=%d, input_point_num=%d, resolution=%f, grid_w=%d, grid_l=%d, grid_h=%d\n", batch_size, input_point_num, resolution, grid_w, grid_l, grid_h);
//    printf("grid_size=%d, blockSize=%d\n", gridSize, blockSize);
    grid_sampling_gpu_kernel<<<gridSize, blockSize>>>(batch_size, input_point_num,
                                                      resolution[0], resolution[1], resolution[2],
                                                      grid_dims[0], grid_dims[1], grid_dims[2],
                                                      input_coors,
                                                      input_num_list,
                                                      input_accu_list,
                                                      output_idx_temp,
                                                      output_num_list,
                                                      grid_buffer);
}
