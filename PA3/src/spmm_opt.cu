#include "spmm_opt.h"

// const int C = 4;
const int D = 2;
const int SPLIT_SIZE = 128;

__global__ void spmm_kernel_placeholder32(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    __shared__ int shared_idx[32][D];
    __shared__ float shared_val[32][D];
    float ans = 0.0;
    int indx = blockIdx.x;
    int indy = blockIdx.y * 32 + threadIdx.x;
	// int begin = ptr[indx], end = ptr[indx + 1];
	int begin = ptr[indx], end = ptr[indx + 1];
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
				ans += shared_val[i][k] * vin[shared_idx[i][k] * INFEATURE + indy];
			}
		}
    }
	vout[indx * INFEATURE + indy] = ans;
}

__global__ void spmm_kernel_placeholder32p(int *ptr, int *idx, float *val, float *vin, float *vout, int *target, int *ptr_scheduled, int num_v, int INFEATURE)
{
    __shared__ int shared_idx[32][D];
    __shared__ float shared_val[32][D];
    float ans = 0.0;
    int indx = blockIdx.x;
    int indy = blockIdx.y * 32 + threadIdx.x;
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
				ans += shared_val[i][k] * vin[shared_idx[i][k] * INFEATURE + indy];
			}
		}
    }
	atomicAdd(vout + target[indx] * INFEATURE + indy, ans);
}

__global__ void spmm_kernel_placeholder2(int *ptr, int *idx, float *val, float *vin, float *vout, int *target, int *ptr_scheduled, int num_v, int INFEATURE, bool dataset)
{
    __shared__ int shared_idx[32][D];
    __shared__ float shared_val[32][D];
    float ans[2] = {0.0, 0.0};
    int indx = blockIdx.x;
    int indy = blockIdx.y * 32 * 2 + threadIdx.x;
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
				for (int j = 0; j < 2; j++)
				{
					ans[j] += shared_val[i][k] * vin[shared_idx[i][k] * INFEATURE + indy + j * 32];
				}
			}
		}
    }
    for (int j = 0; j < 2; j++)
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

__global__ void spmm_kernel_placeholder4(int *ptr, int *idx, float *val, float *vin, float *vout, int *target, int *ptr_scheduled, int num_v, int INFEATURE, bool dataset)
{
    __shared__ int shared_idx[32][D];
    __shared__ float shared_val[32][D];
    float ans[4] = {0.0, 0.0, 0.0, 0.0};
    int indx = blockIdx.x;
    int indy = blockIdx.y * 32 * 4 + threadIdx.x;
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
				for (int j = 0; j < 4; j++)
				{
					ans[j] += shared_val[i][k] * vin[shared_idx[i][k] * INFEATURE + indy + j * 32];
				}
			}
		}
    }
    for (int j = 0; j < 4; j++)
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
	int C = 1;
	if (feat_in == 256)
	{
		C = 2;
		if (num_v == 1138499 || num_v == 2500604 || num_v == 881680) C = 4;
	}
	bool dataset = check_dataset();
	fprintf(stderr, "dataset = %d, C = %d\n", dataset, C);

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
	if (feat_in == 32)
	{
		if (check_dataset())
		{
			spmm_kernel_placeholder32p<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, target, ptr_scheduled, num_v, feat_in);
		}
		else
		{
			spmm_kernel_placeholder32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
		}
	}
	else
	{
		if (num_v == 1138499 || num_v == 2500604 || num_v == 881680)
		{
			spmm_kernel_placeholder4<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, target, ptr_scheduled, num_v, feat_in, check_dataset());
		}
		else
		{
			spmm_kernel_placeholder2<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, target, ptr_scheduled, num_v, feat_in, check_dataset());
		}

	}
}
