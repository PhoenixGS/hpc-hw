// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include <stdio.h>

const int INF = 1000000000;

namespace {
	__global__ void center(int n, int p, int * graph) {
		__shared__ int dis[32][32];
		auto i = p * 32 + threadIdx.x;
		auto j = p * 32 + threadIdx.y;
		if (i < n && j < n) {
			dis[threadIdx.x][threadIdx.y] = graph[i * n + j];
		} else {
			dis[threadIdx.x][threadIdx.y] = INF;
		}
		__syncthreads();
		auto minx = INF;
		for (int k = 0; k < 32; k++) {
			if (threadIdx.x != k && threadIdx.y != k) {
				minx = min(minx, dis[threadIdx.x][k] + dis[k][threadIdx.y]);
			}
		}
		if (i < n && j < n) {
			graph[i * n + j] = min(graph[i * n + j], minx);
		}
	}

	__global__ void cross(int n, int p, int *graph) {
		__shared__ int cent[32][32];
		__shared__ int dis[32][32];
		auto cent_i = p * 32 + threadIdx.x;
		auto cent_j = p * 32 + threadIdx.y;
		if (cent_i < n && cent_j < n) {
			cent[threadIdx.x][threadIdx.y] = graph[cent_i * n + cent_j];
		} else {
			cent[threadIdx.x][threadIdx.y] = INF;
		}
		int base_i, base_j;
		if (blockIdx.x == 0) {
			base_i = p * 32;
			base_j = blockIdx.y * 32;
		} else {
			base_i = blockIdx.y * 32;
			base_j = p * 32;
		}
		auto i = base_i + threadIdx.x;
		auto j = base_j + threadIdx.y;
		if (blockIdx.x == 0) {
			if (cent_i < n && j < n) {
				dis[threadIdx.x][threadIdx.y] = graph[cent_i * n + j];
			} else {
				dis[threadIdx.x][threadIdx.y] = INF;
			}
		} else {
			if (i < n && cent_j < n) {
				dis[threadIdx.x][threadIdx.y] = graph[i * n + cent_j];
			} else {
				dis[threadIdx.x][threadIdx.y] = INF;
			}
		}
		/*if (i < n && j < n) {
			dis[threadIdx.x][threadIdx.y] = graph[i * n + j];
		} else {
			dis[threadIdx.x][threadIdx.y] = INF;
		}*/
		__syncthreads();
		if (blockIdx.y != p) {
			auto minx = INF;
			for (auto t = 0; t < 32; t++) {
				auto k = p * 32 + t;
				if (i != k && j != k) {
					if (blockIdx.x == 0) {
						minx = min(minx, cent[threadIdx.x][t] + dis[t][threadIdx.y]);
					} else {
						minx = min(minx, dis[threadIdx.x][t] + cent[t][threadIdx.y]);
					}
				}
			}
			if (i < n && j < n) {
				graph[i * n + j] = min(graph[i * n + j], minx);
			}
		}
	}

	__global__ void whole(int n, int p, int *graph) {
		__shared__ int cross1[32][32];
		__shared__ int cross2[32][32];
		auto base_i = blockIdx.x * 32;
		auto base_j = blockIdx.y * 32;
		auto cross1_i = blockIdx.x * 32 + threadIdx.x;
		auto cross1_j = p * 32 + threadIdx.y;
		auto cross2_i = p * 32 + threadIdx.x;
		auto cross2_j = blockIdx.y * 32 + threadIdx.y;
		if (cross1_i < n && cross1_j < n) {
			cross1[threadIdx.x][threadIdx.y] = graph[cross1_i * n + cross1_j];
		} else {
			cross1[threadIdx.x][threadIdx.y] = INF;
		}
		if (cross2_i < n && cross2_j < n) {
			cross2[threadIdx.x][threadIdx.y] = graph[cross2_i * n + cross2_j];
		} else {
			cross2[threadIdx.x][threadIdx.y] = INF;
		}
		auto i = base_i + threadIdx.x;
		auto j = base_j + threadIdx.y;
		__syncthreads();
		if (blockIdx.x != p && blockIdx.y != p) {
			auto minx = INF;
			for (auto t = 0; t < 32; t++) {
				auto k = p * 32 + t;
				if (i != k && j != k) {
					minx = min(minx, cross1[threadIdx.x][t] + cross2[t][threadIdx.y]);
				}
			}
			if (i < n && j < n) {
				graph[i * n + j] = min(graph[i * n + j], minx);
			}
		}
	}

	__global__ void kernel(int n, int k, int *graph) {
		auto i = blockIdx.y * blockDim.y + threadIdx.y;
		auto j = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < n && j < n && i != k && j != k) {
			graph[i * n + j] = min(graph[i * n + j], graph[i * n + k] + graph[k * n + j]);
		}
	}

}

void apsp(int n, /* device */ int *graph) {
	for (int p = 0; p < (n - 1) / 32 + 1; p++) {
		dim3 thr(32, 32);
		center<<<1, thr>>>(n, p, graph);
		dim3 blk(2, (n - 1) / 32 + 1);
		cross<<<blk, thr>>>(n, p, graph);
		dim3 blk2((n - 1) / 32 + 1, (n - 1) / 32 + 1);
		whole<<<blk2, thr>>>(n, p, graph);
	}
    /*for (int k = 0; k < n; k++) {
        dim3 thr(32, 32);
        dim3 blk((n - 1) / 32 + 1, (n - 1) / 32 + 1);
        kernel<<<blk, thr>>>(n, k, graph);
    }*/
}

