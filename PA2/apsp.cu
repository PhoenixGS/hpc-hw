// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include <stdio.h>

const int INF = 1000000000;

namespace {
	__device__ int get(int * graph, int n, int i, int j) {
		return (i < n && j < n) ? graph[i * n + j] : INF;
	}

	__device__ void put(int * graph, int n, int i, int j, int minx) {
		if (i >= n) return;
		if (j >= n) return;
		graph[i * n + j] = min(graph[i * n + j], minx);
	}

	__global__ void center(int n, int p, int * graph) {
		__shared__ int dis[32][32];
		auto i = p * 32 + threadIdx.y;
		auto j = p * 32 + threadIdx.x;
		dis[threadIdx.y][threadIdx.x] = get(graph, n, i, j);
		__syncthreads();
		auto minx = INF;
		for (int k = 0; k < 32; k++) {
			minx = min(minx, dis[threadIdx.y][k] + dis[k][threadIdx.x]);
		}
		put(graph, n, i, j, minx);
	}

	__global__ void cross(int n, int p, int *graph) {
		__shared__ int cent[32][32];
		__shared__ int dis[32][32][8];
		auto cent_i = p * 32 + threadIdx.y;
		auto cent_j = p * 32 + threadIdx.x;
		cent[threadIdx.y][threadIdx.x] = get(graph, n, cent_i, cent_j);
		for (int T = 0; T < 8; T++) {
			int base_i, base_j;
			if (blockIdx.y == 0) {
				base_i = p * 32;
				base_j = (blockIdx.x * 8 + T) * 32;
			} else {
				base_i = (blockIdx.x * 8 + T) * 32;
				base_j = p * 32;
			}
			auto i = base_i + threadIdx.y;
			auto j = base_j + threadIdx.x;
			if (blockIdx.y == 0) {
				dis[threadIdx.y][threadIdx.x][T] = get(graph, n, cent_i, j);
			} else {
				dis[threadIdx.y][threadIdx.x][T] = get(graph, n, i, cent_j);
			}
		}
		__syncthreads();
		int minx[8] = {INF, INF, INF, INF, INF, INF, INF, INF};
		for (auto t = 0; t < 32; t++) {
			for (int T = 0; T < 8; T++) {
				if (blockIdx.y == 0) {
					for (auto t = 0; t < 32; t++) {
						minx[T] = min(minx[T], cent[threadIdx.y][t] + dis[t][threadIdx.x][T]);
					}
				} else {
					for (auto t = 0; t < 32; t++) {
						minx[T] = min(minx[T], dis[threadIdx.y][t][T] + cent[t][threadIdx.x]);
					}
				}
			}
		}
		for (int T = 0; T < 8; T++) {
			int base_i, base_j;
			if (blockIdx.y == 0) {
				base_i = p * 32;
				base_j = (blockIdx.x * 8 + T) * 32;
			} else {
				base_i = (blockIdx.x * 8 + T) * 32;
				base_j = p * 32;
			}
			auto i = base_i + threadIdx.y;
			auto j = base_j + threadIdx.x;
			put(graph, n, i, j, minx[T]);
		}
	}

	__global__ void whole(int n, int p, int *graph) {
		__shared__ int cross1[6][32][32];
		__shared__ int cross2[6][32][32];
		for (int T = 0; T < 6; T++)
		{
			auto cross1_i = (blockIdx.y * 6 + T) * 32 + threadIdx.y;
			auto cross1_j = p * 32 + threadIdx.x;
			cross1[T][threadIdx.y][threadIdx.x] = get(graph, n, cross1_i, cross1_j);
		}
		for (int T = 0; T < 6; T++) {
			auto cross2_i = p * 32 + threadIdx.y;
			auto cross2_j = (blockIdx.x * 6 + T) * 32 + threadIdx.x;
			cross2[T][threadIdx.y][threadIdx.x] = get(graph, n, cross2_i, cross2_j);
		}
		__syncthreads();
		for (int T2 = 0; T2 < 6; T2++) {
			for (int T1 = 0; T1 < 6; T1++) {
				auto by = blockIdx.y * 6 + T1;
				auto bx = blockIdx.x * 6 + T2;
				auto base_i = by * 32;
				auto base_j = bx * 32;
				auto i = base_i + threadIdx.y;
				auto j = base_j + threadIdx.x;
				auto minx = INF;
				for (auto t = 0; t < 32; t++) {
					minx = min(minx, cross1[T1][threadIdx.y][t] + cross2[T2][t][threadIdx.x]);
				}
				put(graph, n, i, j, minx);
			}
		}
	}
}

void apsp(int n, /* device */ int *graph) {
	for (int p = 0; p < (n - 1) / 32 + 1; p++) {
		dim3 thr(32, 32);
		center<<<1, thr>>>(n, p, graph);
		dim3 blk((n - 1) / 32 / 8 + 1, 2);
		cross<<<blk, thr>>>(n, p, graph);
		dim3 blk2((n - 1) / 32 / 6 + 1, (n - 1) / 32 / 6 + 1);
		whole<<<blk2, thr>>>(n, p, graph);
	}
}

