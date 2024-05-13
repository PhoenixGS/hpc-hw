// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include <stdio.h>

const int INF = 1000000000;

namespace {
	__global__ void center(int n, int p, int * graph) {
		__shared__ int dis[32][32];
		auto i = p * 32 + threadIdx.y;
		auto j = p * 32 + threadIdx.x;
		if (i < n && j < n) {
			dis[threadIdx.y][threadIdx.x] = graph[i * n + j];
		} else {
			dis[threadIdx.y][threadIdx.x] = INF;
		}
		__syncthreads();
		auto minx = INF;
		for (int k = 0; k < 32; k++) {
			if (threadIdx.y != k && threadIdx.x != k) {
				minx = min(minx, dis[threadIdx.y][k] + dis[k][threadIdx.x]);
			}
		}
		if (i < n && j < n) {
			graph[i * n + j] = min(graph[i * n + j], minx);
		}
	}

	__global__ void cross(int n, int p, int *graph) {
		__shared__ int cent[32][32];
		__shared__ int dis[32][32];
		auto cent_i = p * 32 + threadIdx.y;
		auto cent_j = p * 32 + threadIdx.x;
		if (cent_i < n && cent_j < n) {
			cent[threadIdx.y][threadIdx.x] = graph[cent_i * n + cent_j];
		} else {
			cent[threadIdx.y][threadIdx.x] = INF;
		}
		int base_i, base_j;
		if (blockIdx.y == 0) {
			base_i = p * 32;
			base_j = blockIdx.x * 32;
		} else {
			base_i = blockIdx.x * 32;
			base_j = p * 32;
		}
		auto i = base_i + threadIdx.y;
		auto j = base_j + threadIdx.x;
		if (blockIdx.y == 0) {
			if (cent_i < n && j < n) {
				dis[threadIdx.y][threadIdx.x] = graph[cent_i * n + j];
			} else {
				dis[threadIdx.y][threadIdx.x] = INF;
			}
		} else {
			if (i < n && cent_j < n) {
				dis[threadIdx.y][threadIdx.x] = graph[i * n + cent_j];
			} else {
				dis[threadIdx.y][threadIdx.x] = INF;
			}
		}
		__syncthreads();
		if (blockIdx.x != p) {
			auto minx = INF;
			for (auto t = 0; t < 32; t++) {
				auto k = p * 32 + t;
				if (i != k && j != k) {
					if (blockIdx.y == 0) {
						minx = min(minx, cent[threadIdx.y][t] + dis[t][threadIdx.x]);
					} else {
						minx = min(minx, dis[threadIdx.y][t] + cent[t][threadIdx.x]);
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
		auto base_i = blockIdx.y * 32;
		auto base_j = blockIdx.x * 32;
		auto cross1_i = blockIdx.y * 32 + threadIdx.y;
		auto cross1_j = p * 32 + threadIdx.x;
		auto cross2_i = p * 32 + threadIdx.y;
		auto cross2_j = blockIdx.x * 32 + threadIdx.x;
		if (cross1_i < n && cross1_j < n) {
			cross1[threadIdx.y][threadIdx.x] = graph[cross1_i * n + cross1_j];
		} else {
			cross1[threadIdx.y][threadIdx.x] = INF;
		}
		if (cross2_i < n && cross2_j < n) {
			cross2[threadIdx.y][threadIdx.x] = graph[cross2_i * n + cross2_j];
		} else {
			cross2[threadIdx.y][threadIdx.x] = INF;
		}
		auto i = base_i + threadIdx.y;
		auto j = base_j + threadIdx.x;
		__syncthreads();
		if (blockIdx.y != p && blockIdx.x != p) {
			auto minx = INF;
			for (auto t = 0; t < 32; t++) {
				auto k = p * 32 + t;
				if (i != k && j != k) {
					minx = min(minx, cross1[threadIdx.y][t] + cross2[t][threadIdx.x]);
				}
			}
			if (i < n && j < n) {
				graph[i * n + j] = min(graph[i * n + j], minx);
			}
		}
	}
}

void apsp(int n, /* device */ int *graph) {
	for (int p = 0; p < (n - 1) / 32 + 1; p++) {
		dim3 thr(32, 32);
		center<<<1, thr>>>(n, p, graph);
		dim3 blk((n - 1) / 32 + 1, 2);
		cross<<<blk, thr>>>(n, p, graph);
		dim3 blk2((n - 1) / 32 + 1, (n - 1) / 32 + 1);
		whole<<<blk2, thr>>>(n, p, graph);
		auto time_4 = time(nullptr);
	}
}

