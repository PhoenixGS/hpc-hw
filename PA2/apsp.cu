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
		__shared__ int dis[32][32];
		auto cent_i = p * 32 + threadIdx.y;
		auto cent_j = p * 32 + threadIdx.x;
		cent[threadIdx.y][threadIdx.x] = get(graph, n, cent_i, cent_j);
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
			dis[threadIdx.y][threadIdx.x] = get(graph, n, cent_i, j);
		} else {
			dis[threadIdx.y][threadIdx.x] = get(graph, n, i, cent_j);
		}
		__syncthreads();
		auto minx = INF;
		if (blockIdx.y == 0) {
			for (auto t = 0; t < 32; t++) {
				minx = min(minx, cent[threadIdx.y][t] + dis[t][threadIdx.x]);
			}
		} else {
			for (auto t = 0; t < 32; t++) {
				minx = min(minx, dis[threadIdx.y][t] + cent[t][threadIdx.x]);
			}
		}
		put(graph, n, i, j, minx);
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
		cross1[threadIdx.y][threadIdx.x] = get(graph, n, cross1_i, cross1_j);
		cross2[threadIdx.y][threadIdx.x] = get(graph, n, cross2_i, cross2_j);
		auto i = base_i + threadIdx.y;
		auto j = base_j + threadIdx.x;
		__syncthreads();
		auto minx = INF;
		for (auto t = 0; t < 32; t++) {
			minx = min(minx, cross1[threadIdx.y][t] + cross2[t][threadIdx.x]);
		}
		put(graph, n, i, j, minx);
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

