#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort() {
    /** Your code ... */
    // you can use variables in class Worker: n, nprocs, rank, block_len, data

    int T = 0;
    float* tmp = new float[block_len];
    float* old_data = new float[block_len];

    std::sort(data, data + block_len);

    while (T < nprocs)
    {
        if ((T % 2 == 1) && (rank == 0 || rank == nprocs - 1))
        {
            T++;
            continue;
        }
        int neigh = ((T % 2) == (rank % 2)) ? (rank + 1) : (rank - 1);
        int ps = ((T % 2) == (rank % 2)) ? 1 : -1;

        MPI_Request request[2];
        MPI_Irecv(tmp, block_len, MPI_FLOAT, neigh, 0, MPI_COMM_WORLD, &request[1]);
        memcpy(old_data, data, sizeof(float) * block_len);
        MPI_Isend(old_data, block_len, MPI_FLOAT, neigh, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Wait(&request[1], nullptr);

        if (ps == 1)
        {
            int pt1 = 0, pt2 = 0;
            int now = 0;
            while (now < (int)block_len)
            {
                if (data[pt1] < tmp[pt2])
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
            int pt1 = block_len - 1, pt2 = block_len - 1;
            int now = block_len - 1;
            while (now >= 0)
            {
                if (data[pt1] > tmp[pt2])
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
