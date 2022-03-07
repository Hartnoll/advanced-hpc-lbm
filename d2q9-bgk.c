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

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
  // typedef struct
  // {
  //   float* speeds0;
  //   float* speeds1;
  //   float* speeds2;
  //   float* speeds3;
  //   float* speeds4;
  //   float* speeds5;
  //   float* speeds6;
  //   float* speeds7;
  //   float* speeds8;
  // } t_speed;

/*
** function prototypes
*/

void ptr_swap( float** speeds0, float** speeds1, 
               float** speeds2, float** speeds3, float** speeds4, 
               float** speeds5, float** speeds6, float** speeds7, float** speeds8,
               float** tmp_speeds0, float** tmp_speeds1, float** tmp_speeds2, 
               float** tmp_speeds3, float** tmp_speeds4, float** tmp_speeds5, 
               float** tmp_speeds6, float** tmp_speeds7, float** tmp_speeds8);

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** speeds0, float** speeds1, 
               float** speeds2, float** speeds3, float** speeds4, 
               float** speeds5, float** speeds6, float** speeds7, float** speeds8,
               float** tmp_speeds0, float** tmp_speeds1, float** tmp_speeds2, 
               float** tmp_speeds3, float** tmp_speeds4, float** tmp_speeds5, 
               float** tmp_speeds6, float** tmp_speeds7, float** tmp_speeds8,  
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, float* speeds0, float* speeds1, 
              float* speeds2, float* speeds3, float* speeds4, float* speeds5, 
              float* speeds6, float* speeds7, float* speeds8, float* tmp_speeds0, 
              float* tmp_speeds1, float* tmp_speeds2, float* tmp_speeds3, float* tmp_speeds4, 
              float* tmp_speeds5, float* tmp_speeds6, float* tmp_speeds7, float* tmp_speeds8,  
              int* obstacles);
int accelerate_flow(const t_param params, float* speeds0, float* speeds1, 
              float* speeds2, float* speeds3, float* speeds4, float* speeds5, 
              float* speeds6, float* speeds7, float* speeds8, int* obstacles);
int write_values(const t_param params, float* speeds0, float* speeds1, 
              float* speeds2, float* speeds3, float* speeds4, float* speeds5, 
              float* speeds6, float* speeds7, float* speeds8, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** speeds0, float** speeds1, 
               float** speeds2, float** speeds3, float** speeds4, 
               float** speeds5, float** speeds6, float** speeds7, float** speeds8,
               float** tmp_speeds0, float** tmp_speeds1, float** tmp_speeds2, 
               float** tmp_speeds3, float** tmp_speeds4, float** tmp_speeds5, 
               float** tmp_speeds6, float** tmp_speeds7, float** tmp_speeds8,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, float* speeds0, float* speeds1, 
              float* speeds2, float* speeds3, float* speeds4, float* speeds5, 
              float* speeds6, float* speeds7, float* speeds8);

/* compute average velocity */
float av_velocity(const t_param params, float* speeds0, float* speeds1, 
              float* speeds2, float* speeds3, float* speeds4, float* speeds5, 
              float* speeds6, float* speeds7, float* speeds8, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, float* speeds0, float* speeds1, 
              float* speeds2, float* speeds3, float* speeds4, float* speeds5, 
              float* speeds6, float* speeds7, float* speeds8, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  // t_speed* cells = NULL;
  // t_speed* tmp_cells = NULL;
  float* speeds0 = NULL;
  float* speeds1 = NULL;
  float* speeds2 = NULL; 
  float* speeds3 = NULL;
  float* speeds4 = NULL;
  float* speeds5 = NULL;
  float* speeds6 = NULL;
  float* speeds7 = NULL;
  float* speeds8 = NULL;
  float* tmp_speeds0 = NULL;
  float* tmp_speeds1 = NULL;
  float* tmp_speeds2 = NULL;
  float* tmp_speeds3 = NULL;
  float* tmp_speeds4 = NULL;
  float* tmp_speeds5 = NULL;
  float* tmp_speeds6 = NULL;
  float* tmp_speeds7 = NULL;
  float* tmp_speeds8 = NULL;
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;                                                             /* structure to hold elapsed time */
  double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

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

  /* Total/init time starts here: initialise our data structures and load values from file */
  gettimeofday(&timstr, NULL);
  tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  init_tic=tot_tic;
  initialise(paramfile, obstaclefile, &params, &speeds0, &speeds1, &speeds2, 
            &speeds3, &speeds4, &speeds5, &speeds6, &speeds7, &speeds8, &tmp_speeds0, 
            &tmp_speeds1, &tmp_speeds2, &tmp_speeds3, &tmp_speeds4, &tmp_speeds5, 
            &tmp_speeds6, &tmp_speeds7, &tmp_speeds8, &obstacles, &av_vels);

  /* Init time stops here, compute time starts*/
  gettimeofday(&timstr, NULL);
  init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  comp_tic=init_toc;

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    av_vels[tt] =  timestep(params, speeds0, speeds1, speeds2, 
            speeds3, speeds4, speeds5, speeds6, speeds7, speeds8, tmp_speeds0, 
            tmp_speeds1, tmp_speeds2, tmp_speeds3, tmp_speeds4, tmp_speeds5, 
            tmp_speeds6, tmp_speeds7, tmp_speeds8,  obstacles);
    ptr_swap(&speeds0, &speeds1, &speeds2, 
            &speeds3, &speeds4, &speeds5, &speeds6, &speeds7, &speeds8, &tmp_speeds0, 
            &tmp_speeds1, &tmp_speeds2, &tmp_speeds3, &tmp_speeds4, &tmp_speeds5, 
            &tmp_speeds6, &tmp_speeds7, &tmp_speeds8);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }
  
  /* Compute time stops here, collate time starts*/
  gettimeofday(&timstr, NULL);
  comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  col_tic=comp_toc;

  // Collate data from ranks here 

  /* Total/collate time stops here.*/
  gettimeofday(&timstr, NULL);
  col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  tot_toc = col_toc;
  
  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, speeds0, speeds1, speeds2, 
                                                      speeds3, speeds4, speeds5, speeds6, 
                                                      speeds7, speeds8, obstacles));
  printf("Elapsed Init time:\t\t\t%.6lf (s)\n",    init_toc - init_tic);
  printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
  printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc  - col_tic);
  printf("Elapsed Total time:\t\t\t%.6lf (s)\n",   tot_toc  - tot_tic);
  write_values(params, speeds0, speeds1, speeds2, 
            speeds3, speeds4, speeds5, speeds6, speeds7, speeds8, obstacles, av_vels);
  finalise(&params, &speeds0, &speeds1, &speeds2, 
            &speeds3, &speeds4, &speeds5, &speeds6, &speeds7, &speeds8, &tmp_speeds0, 
            &tmp_speeds1, &tmp_speeds2, &tmp_speeds3, &tmp_speeds4, &tmp_speeds5, 
            &tmp_speeds6, &tmp_speeds7, &tmp_speeds8, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

void ptr_swap(float** speeds0, float** speeds1, 
               float** speeds2, float** speeds3, float** speeds4, 
               float** speeds5, float** speeds6, float** speeds7, float** speeds8,
               float** tmp_speeds0, float** tmp_speeds1, float** tmp_speeds2, 
               float** tmp_speeds3, float** tmp_speeds4, float** tmp_speeds5, 
               float** tmp_speeds6, float** tmp_speeds7, float** tmp_speeds8)
{
  float* tmp = *tmp_speeds0;
  *tmp_speeds0 = *speeds0;
  *speeds0 = tmp; 
  
  float* tmp1 = *tmp_speeds1;
  *tmp_speeds1 = *speeds1;
  *speeds1 = tmp; 
  
  float* tmp2 = *tmp_speeds2;
  *tmp_speeds2 = *speeds2;
  *speeds2 = tmp; 
    
  float* tmp3 = *tmp_speeds3;
  *tmp_speeds3 = *speeds3;
  *speeds3 = tmp; 

  float* tmp4 = *tmp_speeds4;
  *tmp_speeds4 = *speeds4;
  *speeds4 = tmp; 
   
  float* tmp5 = *tmp_speeds5;
  *tmp_speeds5 = *speeds5;
  *speeds5 = tmp;
  
  float* tmp6 = *tmp_speeds6;
  *tmp_speeds6 = *speeds6;
  *speeds6 = tmp; 
    
  float* tmp7 = *tmp_speeds7;
  *tmp_speeds7 = *speeds7;
  *speeds7 = tmp; 

  float* tmp8 = *tmp_speeds8;
  *tmp_speeds8 = *speeds8;
  *speeds8 = tmp; 
}

float timestep(const t_param params, float* restrict speeds0, float* restrict speeds1, 
              float* restrict speeds2, float* restrict speeds3, float* restrict speeds4, 
              float* restrict speeds5, float* restrict speeds6, float* restrict speeds7, 
              float* restrict speeds8, float* restrict tmp_speeds0, float* restrict tmp_speeds1, 
              float* restrict tmp_speeds2, float* restrict tmp_speeds3, float* restrict tmp_speeds4, 
              float* restrict tmp_speeds5, float* restrict tmp_speeds6, float* restrict tmp_speeds7, 
              float* restrict tmp_speeds8, int* restrict obstacles)
{
  accelerate_flow(params, speeds0, speeds1, speeds2, 
            speeds3, speeds4, speeds5, speeds6, speeds7, speeds8, obstacles);

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  const float aw1 = params.density * params.accel / 9.f;
  const float aw2 = params.density * params.accel / 36.f;

  float tot_u = 0.f; 
  int tot_cells = 0;
  __assume((params.nx)%2==0); 
  __assume((params.ny)%2==0);

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      const int y_n = (jj + 1) % params.ny;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      const int accel = (jj == (params.ny - 2)) ? 1 : 0;

      const float speed0 = speeds0[ii + jj*params.nx]; /* central cell, no movement */
      const float speed1 = speeds1[x_w + jj*params.nx]; /* east */
      const float speed2 = speeds2[ii + y_s*params.nx]; /* north */
      const float speed3 = speeds3[x_e + jj*params.nx]; /* west */
      const float speed4 = speeds4[ii + y_n*params.nx]; /* south */
      const float speed5 = speeds5[x_w + y_s*params.nx]; /* north-east */
      const float speed6 = speeds6[x_e + y_s*params.nx]; /* north-west */
      const float speed7 = speeds7[x_e + y_n*params.nx]; /* south-west */
      const float speed8 = speeds8[x_w + y_n*params.nx]; /* south-east */  
      const int isObstacle = obstacles[jj*params.nx + ii];

      /* if the cell contains an obstacle */
      if (isObstacle)
      {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_speeds1[ii + jj*params.nx] = speed3;
        tmp_speeds2[ii + jj*params.nx] = speed4;
        tmp_speeds3[ii + jj*params.nx] = speed1;
        tmp_speeds4[ii + jj*params.nx] = speed2;
        tmp_speeds5[ii + jj*params.nx] = speed7;
        tmp_speeds6[ii + jj*params.nx] = speed8;
        tmp_speeds7[ii + jj*params.nx] = speed5;
        tmp_speeds8[ii + jj*params.nx] = speed6;
      }else
      {
        /* compute local density total */
        const float local_density = (speed0 + speed1 + speed2 + speed3 
                                    + speed4 + speed5 + speed6 + speed7 + speed8);

        const float denom = 1/local_density;
        /* compute x velocity component */
        const float u_x = (speed1
                      + speed5
                      + speed8
                      - (speed3
                        + speed6
                        + speed7))
                      * denom;
        /* compute y velocity component */
        const float u_y = (speed2
                      + speed5
                      + speed6
                      - (speed4
                        + speed7
                        + speed8))
                      * denom;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        const float denom2 = u_sq / (2.f * c_sq);
        /* zero velocity density: weight w0 */
        const float d_equ0 = w0 * local_density
                  * (1.f - denom2);
        /* axis speeds: weight w1 */
        const float u1_div_c_sq = u[1]/c_sq;        
        const float d_equ1 = w1 * local_density * (1.f + u1_div_c_sq
                                        + 0.5f * u1_div_c_sq * u1_div_c_sq
                                        - denom2);
        const float u2_div_c_sq = u[2]/c_sq;        
        const float d_equ2 = w1 * local_density * (1.f + u2_div_c_sq
                                        + 0.5f * u2_div_c_sq * u2_div_c_sq
                                        - denom2);
        const float u3_div_c_sq = u[3]/c_sq;
        const float d_equ3 = w1 * local_density * (1.f + u3_div_c_sq
                                        + 0.5f * u3_div_c_sq * u3_div_c_sq
                                        - denom2);
        const float u4_div_c_sq = u[4]/c_sq;
        const float d_equ4 = w1 * local_density * (1.f + u4_div_c_sq
                                        + 0.5f * u4_div_c_sq * u4_div_c_sq
                                        - denom2);
        /* diagonal speeds: weight w2 */
        const float u5_div_c_sq = u[5]/c_sq;
        const float d_equ5 = w2 * local_density * (1.f + u5_div_c_sq
                                        + 0.5f * u5_div_c_sq * u5_div_c_sq
                                        - denom2);
        const float u6_div_c_sq = u[6]/c_sq;
        const float d_equ6 = w2 * local_density * (1.f + u6_div_c_sq
                                        + 0.5f * u6_div_c_sq * u6_div_c_sq
                                        - denom2);
        const float u7_div_c_sq = u[7]/c_sq;
        const float d_equ7 = w2 * local_density * (1.f + u7_div_c_sq
                                        + 0.5f * u7_div_c_sq * u7_div_c_sq
                                        - denom2);
        const float u8_div_c_sq = u[8]/c_sq;
        const float d_equ8 = w2 * local_density * (1.f + u8_div_c_sq
                                        + 0.5f * u8_div_c_sq * u8_div_c_sq
                                        - denom2);
                                        
        tmp_speeds0[ii + jj*params.nx] = speed0 + params.omega * (d_equ0 - speed0);        
        tmp_speeds1[ii + jj*params.nx] = speed1 + params.omega * (d_equ1 - speed1); //+ aw1 * accel; 
        tmp_speeds2[ii + jj*params.nx] = speed2 + params.omega * (d_equ2 - speed2);                                               
        tmp_speeds3[ii + jj*params.nx] = speed3 + params.omega * (d_equ3 - speed3); //- aw1 * accel;
        tmp_speeds4[ii + jj*params.nx] = speed4 + params.omega * (d_equ4 - speed4);
        tmp_speeds5[ii + jj*params.nx] = speed5 + params.omega * (d_equ5 - speed5); //+ aw2 * accel;
        tmp_speeds6[ii + jj*params.nx] = speed6 + params.omega * (d_equ6 - speed6); //- aw2 * accel;
        tmp_speeds7[ii + jj*params.nx] = speed7 + params.omega * (d_equ7 - speed7); //- aw2 * accel;
        tmp_speeds8[ii + jj*params.nx] = speed8 + params.omega * (d_equ8 - speed8); //+ aw2 * accel;  
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        ++tot_cells;                       
      }
    }
  }
  return tot_u / (float)tot_cells;
}

//Done
int accelerate_flow(const t_param params, float* restrict speeds0, float* restrict speeds1, 
              float* restrict speeds2, float* restrict speeds3, float* restrict speeds4, 
              float* restrict speeds5, float* restrict speeds6, float* restrict speeds7, 
              float* restrict speeds8, int* restrict obstacles)
{
  /* compute weighting factors */
  const float w1 = params.density * params.accel / 9.f;
  const float w2 = params.density * params.accel / 36.f;
  __assume((params.nx)%2==0);
  /* modify the 2nd row of the grid */
  const int jj = params.ny - 2;
  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (speeds3[ii + jj*params.nx] - w1) > 0.f
        && (speeds6[ii + jj*params.nx] - w2) > 0.f
        && (speeds7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      speeds1[ii + jj*params.nx] += w1;
      speeds5[ii + jj*params.nx] += w2;
      speeds8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      speeds3[ii + jj*params.nx] -= w1;
      speeds6[ii + jj*params.nx] -= w2;
      speeds7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

//Done
float av_velocity(const t_param params, float* restrict speeds0, float* restrict speeds1, 
              float* restrict speeds2, float* restrict speeds3, float* restrict speeds4, 
              float* restrict speeds5, float* restrict speeds6, float* restrict speeds7, 
              float* restrict speeds8, int* restrict obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u = 0.f;          /* accumulated magnitudes of velocity for each cell */

  /* loop over all non-blocked cells */
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* ignore occupied cells */
      if (!obstacles[ii + jj*params.nx])
      {
        /* local density total */
        const float local_density = (speeds0[ii + jj*params.nx] 
                                    + speeds1[ii + jj*params.nx]
                                    + speeds2[ii + jj*params.nx]
                                    + speeds3[ii + jj*params.nx]
                                    + speeds4[ii + jj*params.nx]
                                    + speeds5[ii + jj*params.nx]
                                    + speeds6[ii + jj*params.nx]
                                    + speeds7[ii + jj*params.nx]
                                    + speeds8[ii + jj*params.nx]);

        /* x-component of velocity */
        const float u_x = (speeds1[ii + jj*params.nx]
                      + speeds5[ii + jj*params.nx]
                      + speeds8[ii + jj*params.nx]
                      - (speeds3[ii + jj*params.nx]
                         + speeds6[ii + jj*params.nx]
                         + speeds7[ii + jj*params.nx]))
                     / local_density;
        /* compute y velocity component */
        const float u_y = (speeds2[ii + jj*params.nx]
                      + speeds5[ii + jj*params.nx]
                      + speeds6[ii + jj*params.nx]
                      - (speeds4[ii + jj*params.nx]
                         + speeds7[ii + jj*params.nx]
                         + speeds8[ii + jj*params.nx]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

//Done
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, float** speeds0, float** speeds1, 
               float** speeds2, float** speeds3, float** speeds4, 
               float** speeds5, float** speeds6, float** speeds7, float** speeds8,
               float** tmp_speeds0, float** tmp_speeds1, float** tmp_speeds2, 
               float** tmp_speeds3, float** tmp_speeds4, float** tmp_speeds5, 
               float** tmp_speeds6, float** tmp_speeds7, float** tmp_speeds8,  
               int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

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

  /* main grid */
  *speeds0 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*speeds0 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  *speeds1 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*speeds1 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  *speeds2 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*speeds2 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  *speeds3 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*speeds3 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  *speeds4 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*speeds4 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  *speeds5 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*speeds5 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  *speeds6 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*speeds6 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  *speeds7 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*speeds7 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  
  *speeds8 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);

  if (*speeds8 == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_speeds0 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds0 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  *tmp_speeds1 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds1 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  *tmp_speeds2 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds2 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  *tmp_speeds3 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds3 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  *tmp_speeds4 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds4 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  *tmp_speeds5 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds5 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  *tmp_speeds6 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds6 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  *tmp_speeds7 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds7 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);
  
  *tmp_speeds7 = (float*)_mm_malloc(sizeof(float) * (params->ny * params->nx), 64);
  if (*tmp_speeds7 == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  const float w0 = params->density * 4.f / 9.f;
  const float w1 = params->density      / 9.f;
  const float w2 = params->density      / 36.f;
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*speeds0)[ii + jj*params->nx] = w0;
      /* axis directions */
      (*speeds1)[ii + jj*params->nx] = w1;
      (*speeds2)[ii + jj*params->nx] = w1;
      (*speeds3)[ii + jj*params->nx] = w1;
      (*speeds4)[ii + jj*params->nx] = w1;
      /* diagonals */
      (*speeds5)[ii + jj*params->nx] = w2;
      (*speeds6)[ii + jj*params->nx] = w2;
      (*speeds7)[ii + jj*params->nx] = w2;
      (*speeds8)[ii + jj*params->nx] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

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
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** speeds0, float** speeds1, 
              float** speeds2, float** speeds3, float** speeds4, 
              float** speeds5, float** speeds6, float** speeds7, float** speeds8,
              float** tmp_speeds0, float** tmp_speeds1, float** tmp_speeds2, 
              float** tmp_speeds3, float** tmp_speeds4, float** tmp_speeds5, 
              float** tmp_speeds6, float** tmp_speeds7, float** tmp_speeds8,
              int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  _mm_free(*speeds0);
  *speeds0 = NULL;
  _mm_free(*speeds1);
  *speeds1 = NULL;
  _mm_free(*speeds2);
  *speeds2 = NULL;
  _mm_free(*speeds3);
  *speeds3 = NULL;
  _mm_free(*speeds4);
  *speeds4 = NULL;
  _mm_free(*speeds5);
  *speeds5 = NULL;
  _mm_free(*speeds6);
  *speeds6 = NULL;
  _mm_free(*speeds7);
  *speeds7 = NULL;
  _mm_free(*speeds8);
  *speeds8 = NULL;


  _mm_free(*tmp_speeds0);
  *tmp_speeds0 = NULL;
  _mm_free(*tmp_speeds1);
  *tmp_speeds1 = NULL;
  _mm_free(*tmp_speeds2);
  *tmp_speeds2 = NULL;
  _mm_free(*tmp_speeds3);
  *tmp_speeds3 = NULL;
  _mm_free(*tmp_speeds4);
  *tmp_speeds4 = NULL;
  _mm_free(*tmp_speeds5);
  *tmp_speeds5 = NULL;
  _mm_free(*tmp_speeds6);
  *tmp_speeds6 = NULL;
  _mm_free(*tmp_speeds7);
  *tmp_speeds7 = NULL;
  _mm_free(*tmp_speeds8);
  *tmp_speeds8 = NULL;


  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

//Done
float calc_reynolds(const t_param params, float* restrict speeds0, float* restrict speeds1, 
              float* restrict speeds2, float* restrict speeds3, float* restrict speeds4, 
              float* restrict speeds5, float* restrict speeds6, float* restrict speeds7, 
              float* restrict speeds8, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, speeds0, speeds1, speeds2, 
                    speeds3, speeds4, speeds5, speeds6, speeds7, speeds8, obstacles) * params.reynolds_dim / viscosity;
}

//Done
float total_density(const t_param params, float* speeds0, float* speeds1, 
              float* speeds2, float* speeds3, float* speeds4, float* speeds5, 
              float* speeds6, float* speeds7, float* speeds8)
{
  float total = 0.f;
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      total = total + speeds0[ii + jj*params.nx] 
                    + speeds1[ii + jj*params.nx]
                    + speeds2[ii + jj*params.nx]
                    + speeds3[ii + jj*params.nx]
                    + speeds4[ii + jj*params.nx]
                    + speeds5[ii + jj*params.nx]
                    + speeds6[ii + jj*params.nx]
                    + speeds7[ii + jj*params.nx]
                    + speeds8[ii + jj*params.nx];
    }
  }

  return total;
}

//Done
int write_values(const t_param params, float* speeds0, float* speeds1, 
              float* speeds2, float* speeds3, float* speeds4, float* speeds5, 
              float* speeds6, float* speeds7, float* speeds8, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

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
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        const float local_density = (speeds0[ii + jj*params.nx] 
                                    + speeds1[ii + jj*params.nx]
                                    + speeds2[ii + jj*params.nx]
                                    + speeds3[ii + jj*params.nx]
                                    + speeds4[ii + jj*params.nx]
                                    + speeds5[ii + jj*params.nx]
                                    + speeds6[ii + jj*params.nx]
                                    + speeds7[ii + jj*params.nx]
                                    + speeds8[ii + jj*params.nx]);

        /* compute x velocity component */
        u_x = (speeds1[ii + jj*params.nx]
               + speeds5[ii + jj*params.nx]
               + speeds8[ii + jj*params.nx]
               - (speeds3[ii + jj*params.nx]
                  + speeds6[ii + jj*params.nx]
                  + speeds7[ii + jj*params.nx]))
              / local_density;
        /* compute y velocity component */
        u_y = (speeds2[ii + jj*params.nx]
               + speeds5[ii + jj*params.nx]
               + speeds6[ii + jj*params.nx]
               - (speeds4[ii + jj*params.nx]
                  + speeds7[ii + jj*params.nx]
                  + speeds8[ii + jj*params.nx]))
              / local_density;
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

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}