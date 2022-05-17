/*
 ** Code to implement a d2q9-bgk lattice boltzmann scheme.
 ** 'd2' inidates a 2-dimensional grid, and
 ** 'q9' indicates 9 velocities per grid cell.
 ** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
 **
 ** The 'speeds' in each cell are numbered as follows:
 **
 ** 6 2 5
 **  \|/
 ** 3-0-1
 **  /|\
 ** 7 4 8
 **
 ** A 2D grid:
 **
 **           cols
 **       --- --- ---
 **      | D | E | F |
 ** rows  --- --- ---
 **      | A | B | C |
 **       --- --- ---
 **
 ** 'unwrapped' in row major order to give a 1D array:
 **
 **  --- --- --- --- --- ---
 ** | A | B | C | D | E | F |
 **  --- --- --- --- --- ---
 **
 ** Grid indicies are:
 **
 **          ny
 **          ^       cols(ii)
 **          |  ----- ----- -----
 **          | | ... | ... | etc |
 **          |  ----- ----- -----
 ** rows(jj) | | 1,0 | 1,1 | 1,2 |
 **          |  ----- ----- -----
 **          | | 0,0 | 0,1 | 0,2 |
 **          |  ----- ----- -----
 **          ----------------------> nx
 **
 ** Note the names of the input parameter and obstacle files
 ** are passed on the command line, e.g.:
 **
 **   ./d2q9-bgk input.params obstacles.dat
 **
 ** Be sure to adjust the grid dimensions in the parameter file
 ** if you choose a different obstacle file.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>
#include <mpi.h>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int nx;           /* no. of cells in x-direction */
  int ny;           /* no. of cells in y-direction */
  int maxIters;     /* no. of iterations */
  int reynolds_dim; /* dimension for Reynolds number */
  float density;    /* density per link */
  float accel;      /* density redistribution */
  float omega;      /* relaxation parameter */
  int rank;         /* rank of current process */
  int block_size;   /* number of rows in local grid minus the 2 ghost rows.*/
  int size;         /* number of MPI processes. */
  int up_rank;
  int down_rank;
  int remain;
  int startingRow;
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float* restrict speeds0;
  float* restrict speeds1;
  float* restrict speeds2;
  float* restrict speeds3;
  float* restrict speeds4;
  float* restrict speeds5;
  float* restrict speeds6;
  float* restrict speeds7;
  float* restrict speeds8;
} t_speed;


/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed *final_cells_ptr, t_speed *tmp_cells_ptr, t_speed *cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr, float **sendData, float **recvData, float **sendDataLarge, float **recvDataLarge,
               int **recvcounts, int **displs);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed *cells, t_speed *tmp_cells, int *obstacles, float *sendData, float *recvData);
int accelerate_flow(const t_param params, t_speed *cells, int *obstacles, int jj);
int write_values(const t_param params, t_speed *cells, int *obstacles, float *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, t_speed *final_cells_ptr, t_speed *tmp_cells_ptr, t_speed *cells_ptr,
             int **obstacles_ptr, float **av_vels_ptr, float **sendData, float **recvData, float **sendDataLarge, float **recvDataLarge,
             int **recvcounts, int **displs);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed *cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed *cells, int *obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed *cells, int *obstacles);

/* utility functions */
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

float total_cells(int *obstacles, t_param params);

void printrankspeed(t_param* params, t_speed* cells, int rank, char* PrintType);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  char *paramfile = NULL;    /* name of the input parameter file */
  char *obstaclefile = NULL; /* name of a the input obstacle file */
  t_param params;            /* struct to hold parameter values */
  t_speed final_cells;             /* grid containing fluid densities */
  t_speed cells;
  t_speed tmp_cells;                                                                 /* scratch space */
  int *obstacles = NULL;                                                             /* grid indicating which cells are blocked */
  float *av_vels = NULL;                                                             /* a record of the av. velocity computed for each timestep */
  float *sendData = NULL;
  float *recvData = NULL;
  float *sendDataLarge = NULL;
  float *recvDataLarge = NULL;
  int *recvcounts = NULL;
  int *displs = NULL;
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

  // Variables for MPI.
  int rank;
  int size;

  //MPI_Status status;

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }


  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  params.rank = rank;
  params.size = size;

  if (rank == 0) params.down_rank = size - 1;
  else params.down_rank = rank - 1;
  params.up_rank = (rank + 1) % size; 

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic = tot_tic;
  initialise(paramfile, obstaclefile, &params, &final_cells, &tmp_cells, &cells, &obstacles, 
              &av_vels, &sendData, &recvData, &sendDataLarge, &recvDataLarge, &recvcounts, &displs);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic = init_toc;
  const float tot_cells = total_cells(obstacles, params);

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    // if (rank == 0)
    // {
    //   if (tt == 0)
    //   {
    //     printrankspeed(&params, &local_cells, 0, "random.csv");
    //   }
    // }
    float tot_u = timestep(params, &cells, &tmp_cells, obstacles, sendData, recvData);
    if (rank == 0)
    {
      MPI_Reduce(MPI_IN_PLACE, &tot_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      av_vels[tt] = tot_u / tot_cells;
    }
    else
    {
      MPI_Reduce(&tot_u, NULL, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    t_speed temp = tmp_cells;
    tmp_cells = cells;
    cells = temp;
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic = comp_toc;

  // int displs2[params.size];
  // int recvcounts2[params.size];
  // for (int i = 0; i < size; i++)
  // {
  //   if (i < params.ny % size)
  //   {
  //     recvcounts2[i] = ((params.ny/size) + 1) * params.nx;
  //   }
  //   else
  //   {
  //     recvcounts2[i] = (params.ny / size) * params.nx;
  //   }
  //   int dispsum = 0;
  //   for (int a = 0; a < i; a++)
  //   {
  //     dispsum += recvcounts2[a]; 
  //   }
  //   displs2[i] = dispsum;
  // }
  // MPI_Gatherv(&(local_cells.speeds0[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds0, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(&(local_cells.speeds1[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds1, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(&(local_cells.speeds2[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds2, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(&(local_cells.speeds3[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds3, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(&(local_cells.speeds4[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds4, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(&(local_cells.speeds5[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds5, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(&(local_cells.speeds6[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds6, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(&(local_cells.speeds7[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds7, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);
  // MPI_Gatherv(&(local_cells.speeds8[params.nx]), (params.block_size * params.nx), MPI_FLOAT, cells.speeds8, recvcounts2, displs2, MPI_FLOAT, 0, MPI_COMM_WORLD);

  
  if(rank != 0)
  {
    for (int ii = 0; ii < (params.nx * params.block_size); ii++)
    {
      sendDataLarge[ii*9 + 0] = cells.speeds0[params.nx + ii]; 
      sendDataLarge[ii*9 + 1] = cells.speeds1[params.nx + ii];
      sendDataLarge[ii*9 + 2] = cells.speeds1[params.nx + ii];
      sendDataLarge[ii*9 + 3] = cells.speeds1[params.nx + ii];
      sendDataLarge[ii*9 + 4] = cells.speeds1[params.nx + ii];
      sendDataLarge[ii*9 + 5] = cells.speeds1[params.nx + ii];
      sendDataLarge[ii*9 + 6] = cells.speeds1[params.nx + ii];
      sendDataLarge[ii*9 + 7] = cells.speeds1[params.nx + ii];
      sendDataLarge[ii*9 + 8] = cells.speeds1[params.nx + ii];
    }
    MPI_Send(sendDataLarge, (params.nx * params.block_size * 9), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
  else 
  {
    int offset = params.nx * params.block_size;
    for (int src = 1; src < size; src++)
    {
      int recvSize;
      if(params.remain == 0 || src < params.remain) recvSize = (params.nx * params.block_size * 9);
      //else if (src < params.remain) recvSize = (params.nx * params.block_size * 9);
      else recvSize = (params.nx*(params.block_size - 1) *9);
      MPI_Recv(recvDataLarge, recvSize, MPI_FLOAT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      for (int ii =0; ii < recvSize/9; ii++)
      {
        final_cells.speeds0[offset + ii] = recvDataLarge[ii*9 + 0];
        final_cells.speeds1[offset + ii] = recvDataLarge[ii*9 + 1];
        final_cells.speeds2[offset + ii] = recvDataLarge[ii*9 + 2];
        final_cells.speeds3[offset + ii] = recvDataLarge[ii*9 + 3];
        final_cells.speeds4[offset + ii] = recvDataLarge[ii*9 + 4];
        final_cells.speeds5[offset + ii] = recvDataLarge[ii*9 + 5];
        final_cells.speeds6[offset + ii] = recvDataLarge[ii*9 + 6];
        final_cells.speeds7[offset + ii] = recvDataLarge[ii*9 + 7];
        final_cells.speeds8[offset + ii] = recvDataLarge[ii*9 + 8];
      }
      offset += recvSize/9;
    }
    // Place rank 0's local cells into the final cells.
    for (int ii = 0; ii < params.nx * params.block_size; ii++)
    {
      final_cells.speeds0[ii] = cells.speeds0[params.nx + ii];
      final_cells.speeds1[ii] = cells.speeds1[params.nx + ii];
      final_cells.speeds2[ii] = cells.speeds2[params.nx + ii];
      final_cells.speeds3[ii] = cells.speeds3[params.nx + ii];
      final_cells.speeds4[ii] = cells.speeds4[params.nx + ii];
      final_cells.speeds5[ii] = cells.speeds5[params.nx + ii];
      final_cells.speeds6[ii] = cells.speeds6[params.nx + ii];
      final_cells.speeds7[ii] = cells.speeds7[params.nx + ii];
      final_cells.speeds8[ii] = cells.speeds8[params.nx + ii];
    }
  }

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;

  /* write final values and free memory */
  if (rank == 0)
  {
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, &final_cells, obstacles));
    printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
    printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
    printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
    printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
    write_values(params, &final_cells, obstacles, av_vels);
  }
  finalise(&params, &final_cells, &tmp_cells, &cells, &obstacles, &av_vels, &sendData, &recvData, &sendDataLarge, &recvDataLarge, &recvcounts, &displs);
  MPI_Finalize();
  return EXIT_SUCCESS;
}

float total_cells(int *obstacles, t_param params)
{
  float tot_cells = 0;
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.ny; ii++)
    {
      if (!obstacles[ii + jj*params.nx]) tot_cells += 1;
    }
  }
  return tot_cells;
}

int starting_row(t_param* params, int remain)
{
  int starting_row;
  if (params->rank < remain) starting_row = params->rank * params->block_size;
  else starting_row = (params->rank * params->block_size) + remain;
  return starting_row;
  
}

float timestep(const t_param params, t_speed *restrict cells, t_speed *restrict tmp_cells, int *restrict obstacles, float* sendData, float* recvData)
{

  if (params.block_size > 1 && params.rank == (params.size - 1))
    accelerate_flow(params, cells, obstacles, (params.block_size - 1));
  else if (params.block_size == 1 && params.rank == (params.size - 2))
    accelerate_flow(params, cells, obstacles, 1);

  //MPI_Status status;
  // TODO: Halo Exchange goes here
  const int block_size = params.block_size;
  const int down_rank = params.down_rank;
  const int up_rank = params.up_rank;
  for(int ii = 0; ii < params.nx; ii++)
  {
    sendData[(ii*9) + 0] = cells->speeds0[(params.nx * block_size) + ii];
    sendData[(ii*9) + 1] = cells->speeds1[(params.nx * block_size) + ii];
    sendData[(ii*9) + 2] = cells->speeds2[(params.nx * block_size) + ii];
    sendData[(ii*9) + 3] = cells->speeds3[(params.nx * block_size) + ii];
    sendData[(ii*9) + 4] = cells->speeds4[(params.nx * block_size) + ii];
    sendData[(ii*9) + 5] = cells->speeds5[(params.nx * block_size) + ii];
    sendData[(ii*9) + 6] = cells->speeds6[(params.nx * block_size) + ii];
    sendData[(ii*9) + 7] = cells->speeds7[(params.nx * block_size) + ii];
    sendData[(ii*9) + 8] = cells->speeds8[(params.nx * block_size) + ii];
  }
  

  MPI_Sendrecv(sendData, (params.nx*9), MPI_FLOAT, params.up_rank, 0, 
               recvData, (params.nx*9), MPI_FLOAT, params.down_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int ii = 0; ii < params.nx; ii++)
  {
    cells->speeds0[ii] = recvData[ii*9 + 0];
    cells->speeds1[ii] = recvData[ii*9 + 1];
    cells->speeds2[ii] = recvData[ii*9 + 2];
    cells->speeds3[ii] = recvData[ii*9 + 3];
    cells->speeds4[ii] = recvData[ii*9 + 4];
    cells->speeds5[ii] = recvData[ii*9 + 5];
    cells->speeds6[ii] = recvData[ii*9 + 6];
    cells->speeds7[ii] = recvData[ii*9 + 7];
    cells->speeds8[ii] = recvData[ii*9 + 8];
  }
// Start filling in the send data with the bottom grid row.
  for(int ii = 0; ii < params.nx; ii++)
  {
    sendData[ii*9] = cells->speeds0[params.nx + ii];
    sendData[(ii*9) + 1] = cells->speeds1[params.nx + ii];
    sendData[(ii*9) + 2] = cells->speeds2[params.nx + ii];
    sendData[(ii*9) + 3] = cells->speeds3[params.nx + ii];
    sendData[(ii*9) + 4] = cells->speeds4[params.nx + ii];
    sendData[(ii*9) + 5] = cells->speeds5[params.nx + ii];
    sendData[(ii*9) + 6] = cells->speeds6[params.nx + ii];
    sendData[(ii*9) + 7] = cells->speeds7[params.nx + ii];
    sendData[(ii*9) + 8] = cells->speeds8[params.nx + ii];
  }

  MPI_Sendrecv(sendData, (params.nx*9), MPI_FLOAT, params.down_rank, 0, 
               recvData, (params.nx*9), MPI_FLOAT, params.up_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  for (int ii = 0; ii < params.nx; ii++)
  {
    cells->speeds0[(params.nx * (block_size+1)) + ii] = recvData[ii*9];
    cells->speeds1[(params.nx * (block_size+1)) + ii] = recvData[ii*9 + 1];
    cells->speeds2[(params.nx * (block_size+1)) + ii] = recvData[ii*9 + 2];
    cells->speeds3[(params.nx * (block_size+1)) + ii] = recvData[ii*9 + 3];
    cells->speeds4[(params.nx * (block_size+1)) + ii] = recvData[ii*9 + 4];
    cells->speeds5[(params.nx * (block_size+1)) + ii] = recvData[ii*9 + 5];
    cells->speeds6[(params.nx * (block_size+1)) + ii] = recvData[ii*9 + 6];
    cells->speeds7[(params.nx * (block_size+1)) + ii] = recvData[ii*9 + 7];
    cells->speeds8[(params.nx * (block_size+1)) + ii] = recvData[ii*9 + 8];
  }


  // MPI_Sendrecv(&cells->speeds0[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 0,
  //              &cells->speeds0[0], params.nx, MPI_FLOAT, down_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(&cells->speeds1[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 1,
  //              &cells->speeds1[0], params.nx, MPI_FLOAT, down_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(&cells->speeds2[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 2,
  //              &cells->speeds2[0], params.nx, MPI_FLOAT, down_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(&cells->speeds3[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 3,
  //              &cells->speeds3[0], params.nx, MPI_FLOAT, down_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(&cells->speeds4[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 4,
  //              &cells->speeds4[0], params.nx, MPI_FLOAT, down_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(&cells->speeds5[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 5,
  //              &cells->speeds5[0], params.nx, MPI_FLOAT, down_rank, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(&cells->speeds6[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 6,
  //              &cells->speeds6[0], params.nx, MPI_FLOAT, down_rank, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(&cells->speeds7[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 7,
  //              &cells->speeds7[0], params.nx, MPI_FLOAT, down_rank, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // MPI_Sendrecv(&cells->speeds8[block_size * params.nx], params.nx, MPI_FLOAT, up_rank, 8,
  //              &cells->speeds8[0], params.nx, MPI_FLOAT, down_rank, 8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // MPI_Sendrecv(&cells->speeds0[params.nx], params.nx, MPI_FLOAT, down_rank, 9,
  //              &cells->speeds0[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 9, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  // MPI_Sendrecv(&cells->speeds1[params.nx], params.nx, MPI_FLOAT, down_rank, 10,
  //              &cells->speeds1[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  // MPI_Sendrecv(&cells->speeds2[params.nx], params.nx, MPI_FLOAT, down_rank, 11,
  //              &cells->speeds2[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  // MPI_Sendrecv(&cells->speeds3[params.nx], params.nx, MPI_FLOAT, down_rank, 12,
  //              &cells->speeds3[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  // MPI_Sendrecv(&cells->speeds4[params.nx], params.nx, MPI_FLOAT, down_rank, 13,
  //              &cells->speeds4[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  // MPI_Sendrecv(&cells->speeds5[params.nx], params.nx, MPI_FLOAT, down_rank, 14,
  //              &cells->speeds5[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 14, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  // MPI_Sendrecv(&cells->speeds6[params.nx], params.nx, MPI_FLOAT, down_rank, 15,
  //              &cells->speeds6[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 15, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //
  // MPI_Sendrecv(&cells->speeds7[params.nx], params.nx, MPI_FLOAT, down_rank, 16,
  //              &cells->speeds7[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 16, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  //  
  // MPI_Sendrecv(&cells->speeds8[params.nx], params.nx, MPI_FLOAT, down_rank, 17,
  //              &cells->speeds8[(block_size + 1) * params.nx], params.nx, MPI_FLOAT, up_rank, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  float tot_u = 0.f;
  int tot_cells = 0;
  // __assume((params.nx) % 2 == 0);
  // __assume((params.ny) % 2 == 0);
  // __assume_aligned(cells->speeds0, 64);
  // __assume_aligned(cells->speeds1, 64);
  // __assume_aligned(cells->speeds2, 64);
  // __assume_aligned(cells->speeds3, 64);
  // __assume_aligned(cells->speeds4, 64);
  // __assume_aligned(cells->speeds5, 64);
  // __assume_aligned(cells->speeds6, 64);
  // __assume_aligned(cells->speeds7, 64);
  // __assume_aligned(cells->speeds8, 64);
  // __assume_aligned(tmp_cells->speeds0, 64);
  // __assume_aligned(tmp_cells->speeds1, 64);
  // __assume_aligned(tmp_cells->speeds2, 64);
  // __assume_aligned(tmp_cells->speeds3, 64);
  // __assume_aligned(tmp_cells->speeds4, 64);
  // __assume_aligned(tmp_cells->speeds5, 64);
  // __assume_aligned(tmp_cells->speeds6, 64);
  // __assume_aligned(tmp_cells->speeds7, 64);
  // __assume_aligned(tmp_cells->speeds8, 64);

  for (int jj = 1; jj <= params.block_size; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */

      const int y_n = (jj + 1);
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      // const int accel = (jj == (params.ny - 2)) ? 1 : 0;

      const float speed0 = cells->speeds0[ii + jj * params.nx];   /* central cell, no movement */
      const float speed1 = cells->speeds1[x_w + jj * params.nx];  /* east */
      const float speed2 = cells->speeds2[ii + y_s * params.nx];  /* north */
      const float speed3 = cells->speeds3[x_e + jj * params.nx];  /* west */
      const float speed4 = cells->speeds4[ii + y_n * params.nx];  /* south */
      const float speed5 = cells->speeds5[x_w + y_s * params.nx]; /* north-east */
      const float speed6 = cells->speeds6[x_e + y_s * params.nx]; /* north-west */
      const float speed7 = cells->speeds7[x_e + y_n * params.nx]; /* south-west */
      const float speed8 = cells->speeds8[x_w + y_n * params.nx]; /* south-east */
      const int isObstacle = obstacles[(params.startingRow + (jj - 1)) * params.nx + ii];

      /* if the cell contains an obstacle */
      if (isObstacle)
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells->speeds1[ii + jj * params.nx] = speed3;
        tmp_cells->speeds2[ii + jj * params.nx] = speed4;
        tmp_cells->speeds3[ii + jj * params.nx] = speed1;
        tmp_cells->speeds4[ii + jj * params.nx] = speed2;
        tmp_cells->speeds5[ii + jj * params.nx] = speed7;
        tmp_cells->speeds6[ii + jj * params.nx] = speed8;
        tmp_cells->speeds7[ii + jj * params.nx] = speed5;
        tmp_cells->speeds8[ii + jj * params.nx] = speed6;
      }
      else
      {
        /* compute local density total */
        const float local_density = (speed0 + speed1 + speed2 + speed3 + speed4 + speed5 + speed6 + speed7 + speed8);

        const float denom = 1 / local_density;
        /* compute x velocity component */
        const float u_x = (speed1 + speed5 + speed8 - (speed3 + speed6 + speed7)) * denom;
        /* compute y velocity component */
        const float u_y = (speed2 + speed5 + speed6 - (speed4 + speed7 + speed8)) * denom;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */

        const float u1 = u_x;        /* east */
        const float u2 = u_y;        /* north */
        const float u3 = -u_x;       /* west */
        const float u4 = -u_y;       /* south */
        const float u5 = u_x + u_y;  /* north-east */
        const float u6 = -u_x + u_y; /* north-west */
        const float u7 = -u_x - u_y; /* south-west */
        const float u8 = u_x - u_y;  /* south-east */

        /* equilibrium densities */
        /* zero velocity density: weight w0 */
        const float d_equ_0 =
            w0 * local_density * (1.f - u_sq * 1.5f);
        /* axis speeds: weight w1 */
        const float d_equ_1 = w1 * local_density * (1.f + u1 * 3.f + (u1 * u1) * 4.5f - u_sq * 1.5f);
        const float d_equ_2 = w1 * local_density * (1.f + u2 * 3.f + (u2 * u2) * 4.5f - u_sq * 1.5f);
        const float d_equ_3 = w1 * local_density * (1.f + u3 * 3.f + (u3 * u3) * 4.5f - u_sq * 1.5f);
        const float d_equ_4 = w1 * local_density * (1.f + u4 * 3.f + (u4 * u4) * 4.5f - u_sq * 1.5f);
        /* diagonal speeds: weight w2 */
        const float d_equ_5 = w2 * local_density * (1.f + u5 * 3.f + (u5 * u5) * 4.5f - u_sq * 1.5f);
        const float d_equ_6 = w2 * local_density * (1.f + u6 * 3.f + (u6 * u6) * 4.5f - u_sq * 1.5f);
        const float d_equ_7 = w2 * local_density * (1.f + u7 * 3.f + (u7 * u7) * 4.5f - u_sq * 1.5f);
        const float d_equ_8 = w2 * local_density * (1.f + u8 * 3.f + (u8 * u8) * 4.5f - u_sq * 1.5f);

        tmp_cells->speeds0[ii + jj * params.nx] = speed0 + params.omega * (d_equ_0 - speed0);
        tmp_cells->speeds1[ii + jj * params.nx] = speed1 + params.omega * (d_equ_1 - speed1); //+ aw1 * accel;
        tmp_cells->speeds2[ii + jj * params.nx] = speed2 + params.omega * (d_equ_2 - speed2);
        tmp_cells->speeds3[ii + jj * params.nx] = speed3 + params.omega * (d_equ_3 - speed3); //- aw1 * accel;
        tmp_cells->speeds4[ii + jj * params.nx] = speed4 + params.omega * (d_equ_4 - speed4);
        tmp_cells->speeds5[ii + jj * params.nx] = speed5 + params.omega * (d_equ_5 - speed5); //+ aw2 * accel;
        tmp_cells->speeds6[ii + jj * params.nx] = speed6 + params.omega * (d_equ_6 - speed6); //- aw2 * accel;
        tmp_cells->speeds7[ii + jj * params.nx] = speed7 + params.omega * (d_equ_7 - speed7); //- aw2 * accel;
        tmp_cells->speeds8[ii + jj * params.nx] = speed8 + params.omega * (d_equ_8 - speed8); //+ aw2 * accel;
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        tot_cells++;
      }
    }
  }
  // vels_data.tot_cells = (float)tot_cells;
  // vels_data.tot_u = tot_u;
  return tot_u;
}

int accelerate_flow(const t_param params, t_speed *restrict cells, int *restrict obstacles, int jj)
{
  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;
  // __assume((params.nx) % 2 == 0);
  // __assume_aligned(cells->speeds0, 64);
  // __assume_aligned(cells->speeds1, 64);
  // __assume_aligned(cells->speeds2, 64);
  // __assume_aligned(cells->speeds3, 64);
  // __assume_aligned(cells->speeds4, 64);
  // __assume_aligned(cells->speeds5, 64);
  // __assume_aligned(cells->speeds6, 64);
  // __assume_aligned(cells->speeds7, 64);
  // __assume_aligned(cells->speeds8, 64);
  /* modify the 2nd row of the grid */
  //#pragma omp simd
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[(params.startingRow + (jj - 1)) * params.nx + ii] && (cells->speeds3[ii + jj * params.nx] - w1) > 0.f && (cells->speeds6[ii + jj * params.nx] - w2) > 0.f && (cells->speeds7[ii + jj * params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speeds1[ii + jj * params.nx] += w1;
      cells->speeds5[ii + jj * params.nx] += w2;
      cells->speeds8[ii + jj * params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speeds3[ii + jj * params.nx] -= w1;
      cells->speeds6[ii + jj * params.nx] -= w2;
      cells->speeds7[ii + jj * params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed *restrict cells, int *restrict obstacles)
{
  int tot_cells = 0; /* no. of cells used in calculation */
  float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj * params.nx])
      {
        /* compute local density total */
        const float local_density = (cells->speeds0[ii + jj * params.nx] + cells->speeds1[ii + jj * params.nx] + cells->speeds2[ii + jj * params.nx] + cells->speeds3[ii + jj * params.nx] + cells->speeds4[ii + jj * params.nx] + cells->speeds5[ii + jj * params.nx] + cells->speeds6[ii + jj * params.nx] + cells->speeds7[ii + jj * params.nx] + cells->speeds8[ii + jj * params.nx]);

        /* x-component of velocity */
        const float u_x = (cells->speeds1[ii + jj * params.nx] + cells->speeds5[ii + jj * params.nx] + cells->speeds8[ii + jj * params.nx] - (cells->speeds3[ii + jj * params.nx] + cells->speeds6[ii + jj * params.nx] + cells->speeds7[ii + jj * params.nx])) / local_density;
        /* compute y velocity component */
        const float u_y = (cells->speeds2[ii + jj * params.nx] + cells->speeds5[ii + jj * params.nx] + cells->speeds6[ii + jj * params.nx] - (cells->speeds4[ii + jj * params.nx] + cells->speeds7[ii + jj * params.nx] + cells->speeds8[ii + jj * params.nx])) / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed *final_cells_ptr, t_speed *tmp_cells_ptr, t_speed *cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr, float **sendData, 
               float **recvData, float **sendDataLarge, float **recvDataLarge, int **recvcounts, int **displs)
{
  char message[1024]; /* message buffer */
  FILE *fp;           /* file pointer */
  int xx, yy;         /* generic array indices */
  int blocked;        /* indicates whether a cell is blocked by an obstacle */
  int retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1)
    die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1)
    die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1)
    die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1)
    die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1)
    die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1)
    die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1)
    die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */
  int remain = params->ny % params->size;
  params->remain = remain;
  params->block_size = params->ny / params->size;
  if (params->rank < remain) params->block_size += 1;
  params->startingRow = starting_row(params, remain);

  /* main grid */
  if(params->rank == 0)
  {
    final_cells_ptr->speeds0 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
    final_cells_ptr->speeds1 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
    final_cells_ptr->speeds2 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
    final_cells_ptr->speeds3 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
    final_cells_ptr->speeds4 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
    final_cells_ptr->speeds5 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
    final_cells_ptr->speeds6 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
    final_cells_ptr->speeds7 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
    final_cells_ptr->speeds8 = (float*)malloc(sizeof(float) * (params->ny * params->nx));
  }

  cells_ptr->speeds0 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  cells_ptr->speeds1 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  cells_ptr->speeds2 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  cells_ptr->speeds3 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  cells_ptr->speeds4 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  cells_ptr->speeds5 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  cells_ptr->speeds6 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  cells_ptr->speeds7 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  cells_ptr->speeds8 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));

  /* 'helper' grid, used as scratch space */
  tmp_cells_ptr->speeds0 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  tmp_cells_ptr->speeds1 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  tmp_cells_ptr->speeds2 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  tmp_cells_ptr->speeds3 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  tmp_cells_ptr->speeds4 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  tmp_cells_ptr->speeds5 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  tmp_cells_ptr->speeds6 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  tmp_cells_ptr->speeds7 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));
  tmp_cells_ptr->speeds8 = (float*)malloc(sizeof(float) * ((params->block_size + 2) * params->nx));

  // if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL)
    die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density / 9.f;
  float w2 = params->density / 36.f;
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj * params->nx] = 0;
    }
  }
  for (int jj = 1; jj < (params->block_size + 1); jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
        /* centre */
        cells_ptr->speeds0[ii + jj * params->nx] = w0;
        /* axis directions */
        cells_ptr->speeds1[ii + jj * params->nx] = w1;
        cells_ptr->speeds2[ii + jj * params->nx] = w1;
        cells_ptr->speeds3[ii + jj * params->nx] = w1;
        cells_ptr->speeds4[ii + jj * params->nx] = w1;
        /* diagonals */
        cells_ptr->speeds5[ii + jj * params->nx] = w2;
        cells_ptr->speeds6[ii + jj * params->nx] = w2;
        cells_ptr->speeds7[ii + jj * params->nx] = w2;
        cells_ptr->speeds8[ii + jj * params->nx] = w2;
    }
  }
  //if (params->rank == 0) printrankspeed(params, local_cells_ptr, 0, "speed5_vals.csv");
  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3)
      die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1)
      die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1)
      die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1)
      die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy * params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  *recvData = (float*)malloc(sizeof(float) * (params->nx * 9));
  *sendData = (float*)malloc(sizeof(float) * (params->nx * 9));

  *recvDataLarge = (float*)malloc(sizeof(float) * (params->nx * params->block_size * NSPEEDS)); 
  *sendDataLarge = (float*)malloc(sizeof(float) * (params->nx * params->block_size * NSPEEDS));

  // *recvcounts = malloc(sizeof(int) * params->size);
  // *displs = malloc(sizeof(int) * params->size);

  return EXIT_SUCCESS;
}

int finalise(const t_param *params, t_speed *final_cells_ptr, t_speed *tmp_cells_ptr, t_speed *cells_ptr,
             int **obstacles_ptr, float **av_vels_ptr, float **sendData, float **recvData, float **sendDataLarge, float **recvDataLarge,
             int **recvcounts, int **displs)
{
  /*
  ** free up allocated memory
  */
  if (params->rank == 0)
  {
    free(final_cells_ptr->speeds0);
    free(final_cells_ptr->speeds1);
    free(final_cells_ptr->speeds2);
    free(final_cells_ptr->speeds3);
    free(final_cells_ptr->speeds4);
    free(final_cells_ptr->speeds5);
    free(final_cells_ptr->speeds6);
    free(final_cells_ptr->speeds7);
    free(final_cells_ptr->speeds8);
  }

  free(tmp_cells_ptr->speeds0);
  free(tmp_cells_ptr->speeds1);
  free(tmp_cells_ptr->speeds2);
  free(tmp_cells_ptr->speeds3);
  free(tmp_cells_ptr->speeds4);
  free(tmp_cells_ptr->speeds5);
  free(tmp_cells_ptr->speeds6);
  free(tmp_cells_ptr->speeds7);
  free(tmp_cells_ptr->speeds8);

  free(cells_ptr->speeds0);
  free(cells_ptr->speeds1);
  free(cells_ptr->speeds2);
  free(cells_ptr->speeds3);
  free(cells_ptr->speeds4);
  free(cells_ptr->speeds5);
  free(cells_ptr->speeds6);
  free(cells_ptr->speeds7);
  free(cells_ptr->speeds8);

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  free(*sendData);
  *sendData = NULL;

  free(*recvData);
  *recvData = NULL;

  free(*sendDataLarge);
  *sendDataLarge = NULL;

  free(*recvDataLarge);
  *recvDataLarge = NULL;

  // free(*recvcounts);
  // *recvcounts = NULL;

  // free(*displs);
  // *displs = NULL;

  return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed *cells, int *obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed *cells)
{
  float total = 0.f; /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total += cells->speeds0[ii + jj * params.nx];
      total += cells->speeds1[ii + jj * params.nx];
      total += cells->speeds2[ii + jj * params.nx];
      total += cells->speeds3[ii + jj * params.nx];
      total += cells->speeds4[ii + jj * params.nx];
      total += cells->speeds5[ii + jj * params.nx];
      total += cells->speeds6[ii + jj * params.nx];
      total += cells->speeds7[ii + jj * params.nx];
      total += cells->speeds8[ii + jj * params.nx];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed *cells, int *obstacles, float *av_vels)
{
  FILE *fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;          /* per grid cell sum of densities */
  float pressure;               /* fluid pressure in grid cell */
  float u_x;                    /* x-component of velocity in grid cell */
  float u_y;                    /* y-component of velocity in grid cell */
  float u;                      /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj * params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = (cells->speeds0[ii + jj * params.nx] + cells->speeds1[ii + jj * params.nx] + cells->speeds2[ii + jj * params.nx] + cells->speeds3[ii + jj * params.nx] + cells->speeds4[ii + jj * params.nx] + cells->speeds5[ii + jj * params.nx] + cells->speeds6[ii + jj * params.nx] + cells->speeds7[ii + jj * params.nx] + cells->speeds8[ii + jj * params.nx]);

        /* x-component of velocity */
        const float u_x = (cells->speeds1[ii + jj * params.nx] + cells->speeds5[ii + jj * params.nx] + cells->speeds8[ii + jj * params.nx] - (cells->speeds3[ii + jj * params.nx] + cells->speeds6[ii + jj * params.nx] + cells->speeds7[ii + jj * params.nx])) / local_density;
        /* compute y velocity component */
        const float u_y = (cells->speeds2[ii + jj * params.nx] + cells->speeds5[ii + jj * params.nx] + cells->speeds6[ii + jj * params.nx] - (cells->speeds4[ii + jj * params.nx] + cells->speeds7[ii + jj * params.nx] + cells->speeds8[ii + jj * params.nx])) / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char *message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char *exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

void printrankspeed(t_param* params, t_speed* cells, int rank, char* PrintType){

  FILE*   fp;                     /* file pointer */
  fp = fopen(PrintType,"w");
  if (fp == NULL){
      die("The file could mot be open for some reason", __LINE__, __FILE__);
  }

  for (int jj = 1; jj < params->block_size+1; jj++){
      for (int ii = 0; ii < params->nx; ii++)
      {   
          if (ii == 0){
              fprintf(fp,"%.6lf,|,",cells->speeds5[ii + jj*(params->nx)]);
          }else if (ii == params->nx -1){
              fprintf(fp,"|,%.6lf\n",cells->speeds5[ii + jj*(params->nx)]);
          }else {
              fprintf(fp,"%.6lf,",cells->speeds5[ii + jj*(params->nx)]);
          }

      }            
  
    fclose(fp);
  }
}