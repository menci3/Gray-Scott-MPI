#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IDX(i, j, N) ((i) * (N) + (j))

typedef struct {
    
    int processes_per_row;        // Number of processes in X direction (px)
    int processes_per_column;     // Number of processes in Y direction (py)
    
    int my_process_row;           // This process's row in process grid (0 to py-1)
    int my_process_col;           // This process's column in process grid (0 to px-1)
    
    // Size of the local block this process handles
    int local_block_width;        // Number of grid points in X direction
    int local_block_height;       // Number of grid points in Y direction
    
    // Process block start in the global grid
    int global_start_x;           // Global X coordinate of first column
    int global_start_y;           // Global Y coordinate of first row
    
    // MPI ranks of neighboring processes (-1 if no neighbor)
    int north_neighbor_rank;
    int south_neighbor_rank;
    int east_neighbor_rank;
    int west_neighbor_rank;
    
    // MPI data types for efficient communication
    MPI_Datatype column_data_type;  // For sending/receiving columns (vertical boundaries)
    MPI_Datatype row_data_type;     // For sending/receiving rows (horizontal boundaries)
    
} BlockDecomposition;

void setup_block_decomposition(BlockDecomposition* decomp, int global_grid_size, 
                              int my_rank, int total_processes) {
    
    // Arrange processes in a 2D grid
    // px * py = total_processes, with px and py as close as possible
    
    // Start with a square arrangement
    decomp->processes_per_row = (int)sqrt(total_processes);
    
    // Find the largest factor of total_processes -> factor <= sqrt(total_processes)
    while (total_processes % decomp->processes_per_row != 0) {
        decomp->processes_per_row--;
    }
    
    decomp->processes_per_column = total_processes / decomp->processes_per_row;
    
    // Process position in 2D grid
    decomp->my_process_col = my_rank % decomp->processes_per_row;
    decomp->my_process_row = my_rank / decomp->processes_per_row;
    
    // Size of local block
    decomp->local_block_width = global_grid_size / decomp->processes_per_row;
    decomp->local_block_height = global_grid_size / decomp->processes_per_column;
    
    // Check that the grid divides evenly among processes
    if (global_grid_size % decomp->processes_per_row != 0 || 
        global_grid_size % decomp->processes_per_column != 0) {
        if (my_rank == 0) {
            printf("ERROR: Grid size %d must be divisible by both %d and %d\n", 
                   global_grid_size, decomp->processes_per_row, decomp->processes_per_column);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Process block start in the global grid
    decomp->global_start_x = decomp->my_process_col * decomp->local_block_width;
    decomp->global_start_y = decomp->my_process_row * decomp->local_block_height;
    
    // MPI ranks of neighboring processes
    if (decomp->my_process_row > 0) {
        decomp->north_neighbor_rank = my_rank - decomp->processes_per_row;
    } else {
        decomp->north_neighbor_rank = -1;
    }
    
    // South neighbor
    if (decomp->my_process_row < decomp->processes_per_column - 1) {
        decomp->south_neighbor_rank = my_rank + decomp->processes_per_row;
    } else {
        decomp->south_neighbor_rank = -1;
    }
    
    // East neighbor
    if (decomp->my_process_col < decomp->processes_per_row - 1) {
        decomp->east_neighbor_rank = my_rank + 1;
    } else {
        decomp->east_neighbor_rank = -1;
    }
    
    // West neighbor
    if (decomp->my_process_col > 0) {
        decomp->west_neighbor_rank = my_rank - 1;
    } else {
        decomp->west_neighbor_rank = -1;
    }
    
    // MPI derived data types for efficient communication
    
    // One complete row of the local block (contiguous data)
    MPI_Type_contiguous(decomp->local_block_width, MPI_FLOAT, &decomp->row_data_type);
    MPI_Type_commit(&decomp->row_data_type);
    
    // One complete column of the local block (non-contiguous data)
    MPI_Type_vector(decomp->local_block_height, 1, decomp->local_block_width + 2, 
                    MPI_FLOAT, &decomp->column_data_type);
    MPI_Type_commit(&decomp->column_data_type);
}

void initialize(float* U_array, float* V_array, 
                              BlockDecomposition* decomp, int global_grid_size, int my_rank) {
    
    int local_width = decomp->local_block_width;
    int local_height = decomp->local_block_height;
    
    // The local arrays with extra border zones: (local_height + 2) x (local_width + 2)
    
    // Initialize background concentrations for the entire local block
    for (int local_row = 1; local_row <= local_height; local_row++) {
        for (int local_col = 1; local_col <= local_width; local_col++) {
            int array_index = IDX(local_row, local_col, local_width + 2);
            U_array[array_index] = 1.0f;  // High concentration of species U
            V_array[array_index] = 0.0f;  // Low concentration of species V
        }
    }
    
    int perturbation_size = 20;
    int global_center = global_grid_size / 2;
    
    // Global coordinates of the perturbation square
    int global_square_start_x = global_center - perturbation_size / 2;
    int global_square_end_x = global_square_start_x + perturbation_size;
    int global_square_start_y = global_center - perturbation_size / 2;
    int global_square_end_y = global_square_start_y + perturbation_size;
    
    // Convert global square coordinates to local coordinates
    int local_square_start_x = fmax(0, global_square_start_x - decomp->global_start_x);
    int local_square_end_x = fmin(local_width, global_square_end_x - decomp->global_start_x);
    int local_square_start_y = fmax(0, global_square_start_y - decomp->global_start_y);
    int local_square_end_y = fmin(local_height, global_square_end_y - decomp->global_start_y);
    
    // Check if this process's block contains any part of the perturbation square
    if (local_square_start_x < local_square_end_x && local_square_start_y < local_square_end_y) {
        
        // Set different concentrations in the perturbation region
        for (int local_row = local_square_start_y + 1; local_row <= local_square_end_y; local_row++) {
            for (int local_col = local_square_start_x + 1; local_col <= local_square_end_x; local_col++) {
                int array_index = IDX(local_row, local_col, local_width + 2);
                U_array[array_index] = 0.75f;
                V_array[array_index] = 0.25f;
            }
        }
    } else {
        printf("Process %d: No perturbation in my block\n", my_rank);
    }
}

// Function to exchange boundary data between neighboring processes
void exchange_boundary_data(float* U_array, float* V_array, BlockDecomposition* decomp, int my_rank) {
    
    MPI_Request communication_requests[16];  // Maximum 8 sends + 8 receives
    int request_count = 0;
    
    int local_width = decomp->local_block_width;
    int local_height = decomp->local_block_height;
    
    // Communication with north neighbor
    if (decomp->north_neighbor_rank != -1) {
        
        // Send to north
        int my_top_row_start = IDX(1, 1, local_width + 2);
        MPI_Isend(&U_array[my_top_row_start], 1, decomp->row_data_type, 
                  decomp->north_neighbor_rank, 100, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        MPI_Isend(&V_array[my_top_row_start], 1, decomp->row_data_type, 
                  decomp->north_neighbor_rank, 101, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        
        // Receive from north
        int my_top_halo_start = IDX(0, 1, local_width + 2);
        MPI_Irecv(&U_array[my_top_halo_start], 1, decomp->row_data_type, 
                  decomp->north_neighbor_rank, 102, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        MPI_Irecv(&V_array[my_top_halo_start], 1, decomp->row_data_type, 
                  decomp->north_neighbor_rank, 103, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
    }
    
    // Communication with south neighbor
    if (decomp->south_neighbor_rank != -1) {
        
        // Send to south neighbour
        int my_bottom_row_start = IDX(local_height, 1, local_width + 2);
        MPI_Isend(&U_array[my_bottom_row_start], 1, decomp->row_data_type, 
                  decomp->south_neighbor_rank, 102, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        MPI_Isend(&V_array[my_bottom_row_start], 1, decomp->row_data_type, 
                  decomp->south_neighbor_rank, 103, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        
        // Receive from south neighbour
        int my_bottom_halo_start = IDX(local_height + 1, 1, local_width + 2);
        MPI_Irecv(&U_array[my_bottom_halo_start], 1, decomp->row_data_type, 
                  decomp->south_neighbor_rank, 100, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        MPI_Irecv(&V_array[my_bottom_halo_start], 1, decomp->row_data_type, 
                  decomp->south_neighbor_rank, 101, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
    }
    
    
    // Communication with east neighbor
    if (decomp->east_neighbor_rank != -1) {
        
        // Send to east neighbour
        int my_right_col_start = IDX(1, local_width, local_width + 2);
        MPI_Isend(&U_array[my_right_col_start], 1, decomp->column_data_type, 
                  decomp->east_neighbor_rank, 200, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        MPI_Isend(&V_array[my_right_col_start], 1, decomp->column_data_type, 
                  decomp->east_neighbor_rank, 201, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        
        // Receive from east neighbour
        int my_right_halo_start = IDX(1, local_width + 1, local_width + 2);
        MPI_Irecv(&U_array[my_right_halo_start], 1, decomp->column_data_type, 
                  decomp->east_neighbor_rank, 202, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        MPI_Irecv(&V_array[my_right_halo_start], 1, decomp->column_data_type, 
                  decomp->east_neighbor_rank, 203, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
    }
    
    // Communication with west neighbor
    if (decomp->west_neighbor_rank != -1) {
        
        // Send to west neighbour
        int my_left_col_start = IDX(1, 1, local_width + 2);
        MPI_Isend(&U_array[my_left_col_start], 1, decomp->column_data_type, 
                  decomp->west_neighbor_rank, 202, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        MPI_Isend(&V_array[my_left_col_start], 1, decomp->column_data_type, 
                  decomp->west_neighbor_rank, 203, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        
        // Receive from west neighbour
        int my_left_halo_start = IDX(1, 0, local_width + 2);
        MPI_Irecv(&U_array[my_left_halo_start], 1, decomp->column_data_type, 
                  decomp->west_neighbor_rank, 200, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
        MPI_Irecv(&V_array[my_left_halo_start], 1, decomp->column_data_type, 
                  decomp->west_neighbor_rank, 201, MPI_COMM_WORLD, 
                  &communication_requests[request_count++]);
    }
    
    // Wait for all communication to complete before proceeding
    MPI_Waitall(request_count, communication_requests, MPI_STATUSES_IGNORE);
}

void compute_time_step(float* current_U, float* current_V, float* next_U, float* next_V,
                      BlockDecomposition* decomp, float diffusion_U, float diffusion_V, 
                      float feed_rate, float kill_rate, float time_step, int my_rank) {
    
    int local_width = decomp->local_block_width;
    int local_height = decomp->local_block_height;
    
    // Process each point in the local block
    for (int local_row = 1; local_row <= local_height; local_row++) {
        for (int local_col = 1; local_col <= local_width; local_col++) {
            
            int center_index = IDX(local_row, local_col, local_width + 2);
            
            // Get current concentrations at this point
            float current_u = current_U[center_index];
            float current_v = current_V[center_index];
            
            // Compute Laplacian
            int north_index = IDX(local_row - 1, local_col, local_width + 2);
            int south_index = IDX(local_row + 1, local_col, local_width + 2);
            int west_index = IDX(local_row, local_col - 1, local_width + 2);
            int east_index = IDX(local_row, local_col + 1, local_width + 2);
            
            float laplacian_U = current_U[north_index] + current_U[south_index] + 
                               current_U[west_index] + current_U[east_index] - 4.0f * current_u;
            
            float laplacian_V = current_V[north_index] + current_V[south_index] + 
                               current_V[west_index] + current_V[east_index] - 4.0f * current_v;
            
            float reaction_term = current_u * current_v * current_v;

            next_U[center_index] = current_u + time_step * 
                (diffusion_U * laplacian_U - reaction_term + feed_rate * (1.0f - current_u));
            
            next_V[center_index] = current_v + time_step * 
                (diffusion_V * laplacian_V + reaction_term - (feed_rate + kill_rate) * current_v);
        }
    }
}

void gather_and_save_image(int time_step, float* local_V_array, BlockDecomposition* decomp, 
                          int global_grid_size, int my_rank, int total_processes) {
    
    float* global_V_array = NULL;
    
    // Only process 0 needs to allocate memory for the full grid
    if (my_rank == 0) {
        global_V_array = malloc(global_grid_size * global_grid_size * sizeof(float));
    }
    
    int local_width = decomp->local_block_width;
    int local_height = decomp->local_block_height;
    
    MPI_Datatype local_block_without_halo;
    MPI_Type_vector(local_height, local_width, local_width + 2, MPI_FLOAT, &local_block_without_halo);
    MPI_Type_commit(&local_block_without_halo);
    
    MPI_Datatype global_placement_block;
    MPI_Type_vector(local_height, local_width, global_grid_size, MPI_FLOAT, &global_placement_block);
    MPI_Type_commit(&global_placement_block);
    
    if (my_rank == 0) {

        for (int local_row = 0; local_row < local_height; local_row++) {
            for (int local_col = 0; local_col < local_width; local_col++) {
                int local_index = IDX(local_row + 1, local_col + 1, local_width + 2);
                int global_index = local_row * global_grid_size + local_col;
                global_V_array[global_index] = local_V_array[local_index];
            }
        }
        
        // Receive data from all other processes
        for (int source_rank = 1; source_rank < total_processes; source_rank++) {

            // Calculate where this data should go in the global array
            int source_process_col = source_rank % decomp->processes_per_row;
            int source_process_row = source_rank / decomp->processes_per_row;
            int global_start_x = source_process_col * local_width;
            int global_start_y = source_process_row * local_height;
            
            int global_placement_index = global_start_y * global_grid_size + global_start_x;
            
            MPI_Recv(&global_V_array[global_placement_index], 1, global_placement_block,
                     source_rank, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
    } else {
        int local_data_start = IDX(1, 1, local_width + 2);
        MPI_Send(&local_V_array[local_data_start], 1, local_block_without_halo,
                 0, 300, MPI_COMM_WORLD);
    }
    
    MPI_Type_free(&local_block_without_halo);
    MPI_Type_free(&global_placement_block);
    
    if (my_rank == 0) {
        unsigned char* image_data = malloc(global_grid_size * global_grid_size * sizeof(unsigned char));
        
        for (int i = 0; i < global_grid_size * global_grid_size; i++) {
            float concentration = global_V_array[i];
            
            if (concentration < 0.0f) concentration = 0.0f;
            if (concentration > 1.0f) concentration = 1.0f;
            
            image_data[i] = (unsigned char)(concentration * 255.0f);
        }
        
        char filename[128];
        sprintf(filename, "frames_block/frame_%04d.png", time_step / 100);

        stbi_write_png(filename, global_grid_size, global_grid_size, 1, image_data, global_grid_size);
        
        // Clean up
        free(image_data);
        free(global_V_array);
    }
}

int main(int argc, char **argv) {
    
    MPI_Init(&argc, &argv);

    int my_rank, total_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    printf("Process %d of %d: Starting Gray-Scott simulation\n", my_rank, total_processes);

    // Simulation parameters
    int global_grid_size = 128;  // Total grid size
    if (argc >= 2) {
        global_grid_size = atoi(argv[1]);
    }

    const int total_time_steps = 5000;
    const float diffusion_rate_U = 0.16f;
    const float diffusion_rate_V = 0.08f;
    const float feed_rate = 0.060f;
    const float kill_rate = 0.062f;
    const float time_step_size = 1.0f;

    if (my_rank == 0) {
        printf("Simulation parameters:\n");
        printf("  Grid size: %d x %d\n", global_grid_size, global_grid_size);
    }

    // Set up the block decomposition
    BlockDecomposition decomp;
    setup_block_decomposition(&decomp, global_grid_size, my_rank, total_processes);

    if (my_rank == 0) {
        printf("Block-wise distribution: %dx%d processes, local blocks: %dx%d\n", 
               decomp.processes_per_row, decomp.processes_per_column,
               decomp.local_block_width, decomp.local_block_height);
    }

    // Allocate arrays with halo zones (+2 in each dimension)
    int local_size = (decomp.local_block_width + 2) * (decomp.local_block_height + 2);
    float *U = calloc(local_size, sizeof(float));
    float *V = calloc(local_size, sizeof(float));
    float *Unew = calloc(local_size, sizeof(float));
    float *Vnew = calloc(local_size, sizeof(float));

    double total_time = 0.0;
    int repeats = 5;

    for (int r = 0; r < repeats; r++) {
        initialize(U, V, &decomp, global_grid_size, my_rank);

        double start_time = MPI_Wtime();

        for (int t = 0; t < total_time_steps; ++t) {
            exchange_boundary_data(U, V, &decomp, my_rank);
            compute_time_step(U, V, Unew, Vnew, &decomp,
                   diffusion_rate_U, diffusion_rate_V,
                   feed_rate, kill_rate, time_step_size, my_rank);

            float *tmp = U; U = Unew; Unew = tmp;
            tmp = V; V = Vnew; Vnew = tmp;

            if (t > 0 && t % 100 == 0) {
                gather_and_save_image(t, V, &decomp, global_grid_size, my_rank, total_processes);
            }
        }

        double end_time = MPI_Wtime();
        double local_elapsed = end_time - start_time;

        double max_elapsed;
        MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        total_time += max_elapsed;
    }

    if (my_rank == 0) {
        printf("BLOCK - Grid: %dx%d, Cores: %d (%dx%d), Time: %f seconds\n", 
               global_grid_size, global_grid_size, total_processes,
               decomp.processes_per_row, decomp.processes_per_column,
               total_time / repeats);
    }

    // Cleanup
    MPI_Type_free(&decomp.column_data_type);
    MPI_Type_free(&decomp.row_data_type);
    free(U); free(V); free(Unew); free(Vnew);

    MPI_Finalize();
    return 0;
}