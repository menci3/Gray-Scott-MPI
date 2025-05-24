#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void saveFrames(int n, float *U, float *V, int grid_size, int grid_size_whole) {
    unsigned char *img = (unsigned char *)malloc(grid_size_whole);
    for (int i = 0; i < grid_size_whole; i++) {
        float val = V[i];
        if (val < 0) val = 0;
        if (val > 1) val = 1;
        img[i] = (unsigned char)(val * 255);
    }

    char filename[64];
    sprintf(filename, "frames/frame_%04d.png", n/100);
    stbi_write_png(filename, grid_size, grid_size, 1, img, grid_size);
    free(img);
}

void GrayScottSolver(float *U, float *V, float Du, float Dv, float F, float k, float dt, int steps, int grid_size, int grid_size_whole, bool visualize) {

    float *Unew = (float *)malloc(grid_size_whole * sizeof(float));
    float *Vnew = (float *)malloc(grid_size_whole * sizeof(float));

    float lapU, lapV, zmnozek;


    for (int n = 1; n <= steps; n++) {
        for (int i = 1; i < grid_size - 1; i++) {
            for (int j = 1; j < grid_size - 1; j++) {
                int up    = (i - 1 + grid_size) % grid_size;
                int down  = (i + 1) % grid_size;
                int left  = (j - 1 + grid_size) % grid_size;
                int right = (j + 1) % grid_size;

                int index       = i * grid_size + j;
                int index_up    = up * grid_size + j;
                int index_down  = down * grid_size + j;
                int index_left  = i * grid_size + left;
                int index_right = i * grid_size + right;

                lapU = U[index_left] + U[index_right] + U[index_up] + U[index_down] - 4 * U[index];
                lapV = V[index_left] + V[index_right] + V[index_up] + V[index_down] - 4 * V[index];

                zmnozek = U[index] * V[index] * V[index];

                Unew[index] = U[index] + dt * (-zmnozek + F * (1 - U[index]) + Du * lapU);
                Vnew[index] = V[index] + dt * (zmnozek - (F + k) * V[index] + Dv * lapV);
            }
        }

        for (int i = 1; i < grid_size - 1; i++) {
            for (int j = 1; j < grid_size - 1; j++) {
                int index = i * grid_size + j;

                U[index] = Unew[index];
                V[index] = Vnew[index];
            }
        }

        if (visualize && n % 100 == 0) {
            saveFrames(n, U, V, grid_size, grid_size_whole);
        }
    }

    free(Unew);
    free(Vnew);
}


int main(int argc, char *argv[]) {
    // Default parameters
    int grid_size = 0;
    bool visualize = false;  // For saving frames, false when benchmarking

    // Simulation parameters
    float Du = 0.16f;
    float Dv = 0.08f;
    float F  = 0.060f;
    float k  = 0.062f;
    float dt = 1.0f;
    int steps = 5000;

    // Parse command line arguments
    if (argc < 2) {
        printf("USAGE: %s grid_size\n", argv[0]);
        printf("  grid_size: Size of the simulation grid (NxN)\n");
        return 1;
    }

    // Parse grid size (first argument)
    grid_size = atoi(argv[1]);
    if (grid_size <= 0) {
        fprintf(stderr, "Invalid grid size: %d\n", grid_size);
        return 1;
    }

    int grid_size_whole = grid_size * grid_size;
    printf("Grid size: %d x %d\n", grid_size, grid_size);

    // Allocate memory
    float *U = (float *)malloc(grid_size_whole * sizeof(float));
    float *V = (float *)malloc(grid_size_whole * sizeof(float));


    if (!U || !V) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    int repeats = 5;
    float total_time = 0.0;

    for (int r = 0; r < repeats; r++) {
        // Initialize grid
        memset(U, 1, grid_size_whole);
        memset(V, 0, grid_size_whole);

        for (int i = grid_size / 2 - 10; i < grid_size / 2 + 10; i++) {
            for (int j = grid_size / 2 - 10; j < grid_size / 2 + 10; j++) {
                int index = i * grid_size + j;
                U[index] = 0.75;
                V[index] = 0.25;
            }
        }

        clock_t begin = clock();

        GrayScottSolver(U, V, Du, Dv, F, k, dt, steps, grid_size, grid_size_whole, visualize);

        clock_t end = clock();
        total_time += ((float)(end - begin) / CLOCKS_PER_SEC);

        printf("nth time: %.3f seconds\n", ((float)(end - begin) / CLOCKS_PER_SEC));
    }


    printf("Sequential method time: %.3f seconds\n", total_time / repeats);

    free(U);
    free(V);
    
    return 0;
}