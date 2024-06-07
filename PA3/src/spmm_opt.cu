#include "spmm_opt.h"

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    __shared__ int shared_idx[32];
    __shared__ float shared_val[32];
    float ans = 0.0;
    int indx = blockIdx.x;
    int indy = blockIdx.y * 32 + threadIdx.x;
    int begin = ptr[indx], end = ptr[indx + 1];
    for (int now = begin; now < end; now += 32)
    {
        if (now + threadIdx.x < end)
        {
            shared_idx[threadIdx.x] = idx[now + threadIdx.x];
            shared_val[threadIdx.x] = val[now + threadIdx.x];
        }
        __syncthreads();
        for (int i = 0; i < 32 && now + i < end; i++)
        {
            ans += shared_val[i] * vin[shared_idx[i] * INFEATURE + indy];
        }
    }
    vout[indx * INFEATURE + indy] = ans;
}
void SpMMOpt::preprocess(float *vin, float *vout)
{
    int WARP_SIZE = 32;
    grid.x = num_v;
    grid.y = (feat_in + WARP_SIZE - 1) / WARP_SIZE;
    block.x = WARP_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
    spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
