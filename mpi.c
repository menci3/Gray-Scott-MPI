#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IDX(i, j, N) ((i) * (N) + (j))


void initialize(float *U, float *V, int local_rows, int N, int rank) {
    for (int i = 1; i <= local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            U[IDX(i, j, N)] = 1.0f;
            V[IDX(i, j, N)] = 0.0f;
        }
    }

    int square_size = 20;
    int global_mid = N / 2;

    int square_start_i = global_mid - square_size / 2;
    int square_end_i = square_start_i + square_size;
    int square_start_j = global_mid - square_size / 2;
    int square_end_j = square_start_j + square_size;

    // Determine the global start row of this rank
    int global_row_start = rank * local_rows;

    for (int local_i = 1; local_i <= local_rows; ++local_i) {
        int global_i = global_row_start + local_i - 1;

        if (global_i >= square_start_i && global_i < square_end_i) {
            for (int j = square_start_j; j < square_end_j; ++j) {
                if (j >= 0 && j < N) {
                    U[IDX(local_i, j, N)] = 0.75f;
                    V[IDX(local_i, j, N)] = 0.25f;
                }
            }
        }
    }
}


void gather_and_save(int step, float *V_local, int local_rows, int N, int rank, int size) {
    // Buffer on rank 0 to hold the full V grid
    float *V_full = NULL;

    if (rank == 0) {
        V_full = malloc(N * N * sizeof(float));
    }

    // Each process sends only the valid rows (skip halo rows)
    float *V_sendbuf = malloc(local_rows * N * sizeof(float));
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            V_sendbuf[i * N + j] = V_local[IDX(i + 1, j, N)];  // skip halo row 0
        }
    }

    // Gather the parts on rank 0
    MPI_Gather(V_sendbuf, local_rows * N, MPI_FLOAT,
               V_full, local_rows * N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    free(V_sendbuf);

    if (rank == 0) {
        // Convert float values [0,1] to unsigned char [0,255]
        unsigned char *img = malloc(N * N * sizeof(unsigned char));
        for (int i = 0; i < N * N; i++) {
            float val = V_full[i];
            if (val < 0) val = 0;
            if (val > 1) val = 1;
            img[i] = (unsigned char)(val * 255);
        }

        char filename[64];
        sprintf(filename, "frames/frame_%04d.png", step/100);

        stbi_write_png(filename, N, N, 1, img, N);

        free(img);
        free(V_full);
    }
}


void exchange_borders(float *U, float *V, int N, int local_rows, int rank, int size) {
    MPI_Status status;

    // Top (send row 1, receive row 0)
    if (rank > 0) {
        MPI_Sendrecv(&U[IDX(1, 0, N)], N, MPI_FLOAT, rank - 1, 0,
                     &U[IDX(0, 0, N)], N, MPI_FLOAT, rank - 1, 1,
                     MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&V[IDX(1, 0, N)], N, MPI_FLOAT, rank - 1, 2,
                     &V[IDX(0, 0, N)], N, MPI_FLOAT, rank - 1, 3,
                     MPI_COMM_WORLD, &status);
    }

    // Bottom (send row local_rows, receive row local_rows+1)
    if (rank < size - 1) {
        MPI_Sendrecv(&U[IDX(local_rows, 0, N)], N, MPI_FLOAT, rank + 1, 1,
                     &U[IDX(local_rows + 1, 0, N)], N, MPI_FLOAT, rank + 1, 0,
                     MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&V[IDX(local_rows, 0, N)], N, MPI_FLOAT, rank + 1, 3,
                     &V[IDX(local_rows + 1, 0, N)], N, MPI_FLOAT, rank + 1, 2,
                     MPI_COMM_WORLD, &status);
    }
}

void update(float *U, float *V, float *Unew, float *Vnew,
            int local_rows, int N, float Du, float Dv, float F, float k, float dt) {

    for (int i = 1; i <= local_rows; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            float u = U[IDX(i, j, N)];
            float v = V[IDX(i, j, N)];

            float lapU = U[IDX(i-1, j, N)] + U[IDX(i+1, j, N)] +
                         U[IDX(i, j-1, N)] + U[IDX(i, j+1, N)] - 4 * u;

            float lapV = V[IDX(i-1, j, N)] + V[IDX(i+1, j, N)] +
                         V[IDX(i, j-1, N)] + V[IDX(i, j+1, N)] - 4 * v;

            float uvv = u * v * v;
            Unew[IDX(i, j, N)] = u + dt * (Du * lapU - uvv + F * (1.0f - u));
            Vnew[IDX(i, j, N)] = v + dt * (Dv * lapV + uvv - (F + k) * v);
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 128; // Grid size
    
    if (argc >= 2) {
        N = atoi(argv[1]);
    }

    const int steps = 5000;
    const float Du = 0.16f, Dv = 0.08f, F = 0.060f, k = 0.062f, dt = 1.0f;

    // Row-wise partitioning
    int local_rows = N / size;
    if (N % size != 0) {
        if (rank == 0) printf("Grid size must be divisible by number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // +2 for halo rows
    float *U = calloc((local_rows + 2) * N, sizeof(float));
    float *V = calloc((local_rows + 2) * N, sizeof(float));
    float *Unew = calloc((local_rows + 2) * N, sizeof(float));
    float *Vnew = calloc((local_rows + 2) * N, sizeof(float));

    double total_time = 0.0;
    int repeats = 5;

    for (int r = 0; r < repeats; r++) {
        initialize(U, V, local_rows, N, rank);

        double start_time = MPI_Wtime();

        for (int t = 0; t < steps; ++t) {
            exchange_borders(U, V, N, local_rows, rank, size);
            update(U, V, Unew, Vnew, local_rows, N, Du, Dv, F, k, dt);

            float *tmp = U; U = Unew; Unew = tmp;
            tmp = V; V = Vnew; Vnew = tmp;

            if (t > 0 && t % 100 == 0) {
                gather_and_save(t, V, local_rows, N, rank, size);
            }

        }

        double end_time = MPI_Wtime();
        double local_elapsed = end_time - start_time;

        double max_elapsed;
        MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        total_time += max_elapsed;
    }


    if (rank == 0) {
        printf("Grid: %dx%d, Cores: %d, Time: %f seconds\n", N, N, size, total_time / repeats);
    }

    free(U); free(V); free(Unew); free(Vnew);

    MPI_Finalize();
    return 0;
}
