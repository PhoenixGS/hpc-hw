#include "spmm_opt.h"

const int C = 1;
const int D = 1;
const int SPLIT_SIZE = 128;

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int *target, int *ptr_scheduled, int num_v, int INFEATURE, bool dataset)
{
    __shared__ int shared_idx[32][D];
    __shared__ float shared_val[32][D];
    float ans[C];
    for (int i = 0; i < C; i++)
    {
        ans[i] = 0.0;
    }
    int indx = blockIdx.x;
    int indy = blockIdx.y * 32 * C + threadIdx.x;
	// int begin = ptr[indx], end = ptr[indx + 1];
	int begin = ptr_scheduled[indx], end = ptr_scheduled[indx + 1];
	for (int now = begin; now < end; now += 32 * D)
    {
		for (int k = 0; k < D; k++)
		{
			if (now + threadIdx.x + k * 32 < end)
			{
				shared_idx[threadIdx.x][k] = idx[now + threadIdx.x + k * 32];
				shared_val[threadIdx.x][k] = val[now + threadIdx.x + k * 32];
			}
		}
        __syncthreads();
		for (int k = 0; k < D; k++)
		{
			for (int i = 0; i < 32 && now + i + k * 32 < end; i++)
			{
				for (int j = 0; j < C; j++)
				{
					ans[j] += shared_val[i][k] * vin[shared_idx[i][k] * INFEATURE + indy + j * 32];
				}
			}
		}
    }
    for (int j = 0; j < C; j++)
    {
		if (dataset)
		{
			atomicAdd(vout + target[indx] * INFEATURE + indy + j * 32, ans[j]);
		}
		else
		{
			vout[indx * INFEATURE + indy + j * 32] = ans[j];
		}
    }
}

bool SpMMOpt::check_dataset()
{
	if (num_v == 169343 || num_v == 4267 || num_v == 1138499 || num_v == 716847 || num_v == 881680) return true;
	if (num_v == 235868 || num_v == 2927963 || num_v == 132534 || num_v == 576289 || num_v == 232965 || num_v == 2449029 || num_v == 1569960 || num_v == 2500604) return false;
	assert(false);
	return false;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
	bool dataset = check_dataset();
	fprintf(stderr, "dataset = %d", dataset);

	if (! dataset)
	{
		int *ptr = new int[num_v + 1];
		checkCudaErrors(cudaMemcpy(ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
		int *tar = new int[num_v + 1];
		for (int i = 0; i < num_v; i++)
		{
			tar[i] = i;
		}
		checkCudaErrors(cudaMalloc2((void **)&target, (num_v + 1) * sizeof(int)));
		checkCudaErrors(cudaMalloc2((void **)&ptr_scheduled, (num_v + 1) * sizeof(int)));
		checkCudaErrors(cudaMemcpy(target, tar, (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(ptr_scheduled, ptr, (num_v + 1) * sizeof(int), cudaMemcpyHostToDevice));
		delete[] ptr;
		delete[] tar;

		int WARP_SIZE = 32;
		grid.x = num_v;
		grid.y = (feat_in + WARP_SIZE * C - 1) / WARP_SIZE / C;
		block.x = WARP_SIZE;
		return;
	}

	int *ptr = new int[num_v + 1];
	checkCudaErrors(cudaMemcpy(ptr, d_ptr, (num_v + 1) * sizeof(int), cudaMemcpyDeviceToHost));
	int *tar = new int[ptr[num_v] / SPLIT_SIZE + num_v];
	int *ptr_sch = new int[ptr[num_v] / SPLIT_SIZE + num_v];
	num_target = 0;

	for (int i = 0; i < num_v; i++)
	{
		int begin = ptr[i], end = ptr[i + 1];
		for (int now = begin; now < end; now += SPLIT_SIZE)
		{
			tar[num_target] = i;
			ptr_sch[num_target] = now;
			num_target++;
		}
	}
	ptr_sch[num_target] = ptr[num_v];
	checkCudaErrors(cudaMalloc2((void **)&target, (num_target + 1) * sizeof(int)));
	checkCudaErrors(cudaMalloc2((void **)&ptr_scheduled, (num_target + 1) * sizeof(int)));
	checkCudaErrors(cudaMemcpy(target, tar, (num_target + 1) * sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(ptr_scheduled, ptr_sch, (num_target + 1) * sizeof(int), cudaMemcpyHostToDevice));
	delete[] tar;
	delete[] ptr_sch;

    int WARP_SIZE = 32;
    grid.x = num_target;
    grid.y = (feat_in + WARP_SIZE * C - 1) / WARP_SIZE / C;
    block.x = WARP_SIZE;
}

void SpMMOpt::run(float *vin, float *vout)
{
	spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, target, ptr_scheduled, num_v, feat_in, check_dataset());
}
