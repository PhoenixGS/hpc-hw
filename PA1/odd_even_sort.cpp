#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

void Worker::sort() {
    /** Your code ... */
    // you can use variables in class Worker: n, nprocs, rank, block_len, data

    size_t max_size = ceiling(n, nprocs); // max size of each block
    int T = 0;
    float* tmp = new float[max_size]; // data from neighbour
    float* new_data = new float[block_len]; // merge data and tmp

    std::sort(data, data + block_len); // sort in each block

    MPI_Request request[2];
    MPI_Status status;
    int neigh_len;
    while (T < nprocs)
    {
        int neigh = ((T % 2) == (rank % 2)) ? (rank + 1) : (rank - 1);
        if (neigh < 0 || neigh >= nprocs)
        {
            T++;
            continue;
        }
        int ps = ((T % 2) == (rank % 2)) ? 1 : -1;

        MPI_Irecv(tmp, max_size, MPI_FLOAT, neigh, T, MPI_COMM_WORLD, &request[1]); // receive data from neighbour
        MPI_Isend(data, block_len, MPI_FLOAT, neigh, T, MPI_COMM_WORLD, &request[0]); // send merged data
        MPI_Wait(&request[1], &status); 
        MPI_Get_count(&status, MPI_FLOAT, &neigh_len); // get length of neighbour's data

        if (ps == 1) // merge two sorted sequence
        {
            int pt1 = 0, pt2 = 0;
            int now = 0;
            while (now < (int)block_len)
            {
                if (pt2 == neigh_len || data[pt1] < tmp[pt2])
                {
                    new_data[now] = data[pt1];
                    pt1++;
                }
                else
                {
                    new_data[now] = tmp[pt2];
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
                    new_data[now] = data[pt1];
                    pt1--;
                }
                else
                {
                    new_data[now] = tmp[pt2];
                    pt2--;
                }
                now--;
            }
        }
        T++;
        MPI_Wait(&request[0], nullptr);
        std::swap(data, new_data);
    }
    // for (int i = 0; i < (int)block_len; i++)
    // {
    //     printf("%d %d %f\n", rank, i, data[i]);
    // }
    delete[] tmp;
    delete[] new_data;
}
