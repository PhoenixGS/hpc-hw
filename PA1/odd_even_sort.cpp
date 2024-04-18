#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort() {
    /** Your code ... */
    // you can use variables in class Worker: n, nprocs, rank, block_len, data

    size_t block_size = ceiling(n, nprocs);
    int T = 0;
    float* tmp = new float[block_size];
    float* old_data = new float[block_len];

    std::sort(data, data + block_len);

    while (T < nprocs)
    {
        int neigh = ((T % 2) == (rank % 2)) ? (rank + 1) : (rank - 1);
        if (neigh < 0 || neigh >= nprocs)
        {
            T++;
            continue;
        }
        int ps = ((T % 2) == (rank % 2)) ? 1 : -1;

        MPI_Request request[2];
        MPI_Status status;
        MPI_Irecv(tmp, block_size, MPI_FLOAT, neigh, T, MPI_COMM_WORLD, &request[1]);
        memcpy(old_data, data, sizeof(float) * block_len);
        MPI_Isend(old_data, block_len, MPI_FLOAT, neigh, T, MPI_COMM_WORLD, &request[0]);
        MPI_Wait(&request[1], &status);
        int neigh_len;
        MPI_Get_count(&status, MPI_FLOAT, &neigh_len);

        if (ps == 1)
        {
            int pt1 = 0, pt2 = 0;
            int now = 0;
            while (now < (int)block_len)
            {
                if (pt2 == neigh_len || data[pt1] < tmp[pt2])
                {
                    data[now] = data[pt1];
                    pt1++;
                }
                else
                {
                    data[now] = tmp[pt2];
                    pt2++;
                }
                now++;
            }
        }
        else
        {
            int pt1 = block_len - 1, pt2 = neigh_len - 1;
            int now = block_len - 1;
            while (now >= 0)
            {
                if (pt2 < 0 || data[pt1] > tmp[pt2])
                {
                    data[now] = data[pt1];
                    pt1--;
                }
                else
                {
                    data[now] = tmp[pt2];
                    pt2--;
                }
                now--;
            }
        }
        T++;
        MPI_Wait(&request[0], nullptr);
    }
    delete[] tmp;
    delete[] old_data;
}
