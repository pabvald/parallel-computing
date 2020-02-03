/*
 * Simplified simulation of fire extinguishing
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2018/2019
 *
 * v1.4
 *
 * (c) 2019 Arturo Gonzalez Escribano
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "cputils.h"
#include <cuda.h>

#define RADIUS_TYPE_1		3
#define RADIUS_TYPE_2_3		9
#define THRESHOLD	0.1f

/* Structure to store data of an extinguishing team */
typedef struct {
	int x,y;
	int type;
	int target;
} Team;

/* Structure to store data of a fire focal point */
typedef struct {
	int x,y;
	int start;
	int heat;
	int active; // States: 0 Not yet activated; 1 Active; 2 Deactivated by a team
} FocalPoint;

/* Macro function to simplify accessing with two coordinates to a flattened array */
#define accessMat( arr, exp1, exp2 )	arr[ (exp1) * paddingColumns + (exp2) ]

/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s <config_file> | <command_line_args>\n", program_name );
	fprintf(stderr,"\t<config_file> ::= -f <file_name>\n");
	fprintf(stderr,"\t<command_line_args> ::= <rows> <columns> <maxIter> <numTeams> [ <teamX> <teamY> <teamType> ... ] <numFocalPoints> [ <focalX> <focalY> <focalStart> <focalTemperature> ... ]\n");
	fprintf(stderr,"\n");
}

#ifdef DEBUG
/* 
 * Function: Print the current state of the simulation 
 */
void print_status( int iteration, int rows, int columns, float *surface, int num_teams, Team *teams, int num_focal, FocalPoint *focal, float global_residual ) {
	/* 
	 * You don't need to optimize this function, it is only for pretty printing and debugging purposes.
	 * It is not compiled in the production versions of the program.
	 * Thus, it is never used when measuring times in the leaderboard
	 */
	int i,j;

	printf("Iteration: %d\n", iteration );
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	for( i=0; i<rows; i++ ) {
		printf("|");
		for( j=0; j<columns; j++ ) {
			char symbol;
			if ( accessMat( surface, i, j ) >= 1000 ) symbol = '*';
			else if ( accessMat( surface, i, j ) >= 100 ) symbol = '0' + (int)(accessMat( surface, i, j )/100);
			else if ( accessMat( surface, i, j ) >= 50 ) symbol = '+';
			else if ( accessMat( surface, i, j ) >= 25 ) symbol = '.';
			else symbol = '0';

			int t;
			int flag_team = 0;
			for( t=0; t<num_teams; t++ ) 
				if ( teams[t].x == i && teams[t].y == j ) { flag_team = 1; break; }
			if ( flag_team ) printf("[%c]", symbol );
			else {
				int f;
				int flag_focal = 0;
				for( f=0; f<num_focal; f++ ) 
					if ( focal[f].x == i && focal[f].y == j && focal[f].active == 1 ) { flag_focal = 1; break; }
				if ( flag_focal ) printf("(%c)", symbol );
				else printf(" %c ", symbol );
			}
		}
		printf("|\n");
	}
	printf("+");
	for( j=0; j<columns; j++ ) printf("---");
	printf("+\n");
	printf("Global residual: %f\n\n", global_residual);
}
#endif



/**************************************************************** ERROR CHECKING **************************************************************************/

/**
 * Check if an error has ocurred, in which case print the error
 * and a given message, and exit.
 * @param err - cuda error
 * @param s - message (indicating where the error ocurred)
 */
 void CHECK_ERROR(const char* s) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("--ERROR (%s): %s\n", s, cudaGetErrorString(err));
		exit( EXIT_FAILURE );
	}
}



/******************************************************************* KERNELS *****************************************************************************/

/**
 * Kernel 1
 * Update the surface(skip borders)
 *
 */
__global__ void propagate_kernel(float *surface, float *surfaceCopy, int rows, int columns, int paddingColumns) {

	int gid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);

	if (gid >= rows*paddingColumns) return;

	int i = gid / paddingColumns;
	int j = gid % paddingColumns;

	if (i <= 0 || i >= rows-1 || j <= 0 || j >= columns-1) return; // Out of heated surface 

	accessMat( surface, i, j ) = ( 
									accessMat( surfaceCopy, i-1, j ) +
									accessMat( surfaceCopy, i+1, j ) +
									accessMat( surfaceCopy, i, j-1 ) +
									accessMat( surfaceCopy, i, j+1 ) ) / 4;
}

/**
 * Kernel 2
 * Compute the residual difference (absolute value).
 */
__global__ void difference_kernel(float *global_residual, float *surface, float * surfaceCopy, int rows, int columns, int paddingColumns) {

	int gid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);

	if (gid >= rows*paddingColumns) return;

	int i = gid / paddingColumns;
	int j = gid % paddingColumns;

	if (i <= 0 || i >= rows-1 || j <= 0 || j >= columns-1) {
		accessMat( global_residual, i, j ) = 0;
		return; // Out of heated surface 
	} 
	accessMat( global_residual, i, j ) = fabs(accessMat( surface, i, j) - accessMat( surfaceCopy, i, j ));
}

/**
 * Kernel 3
 * Get the maximum value in data
 */ 
__global__ void reduce_max1_kernel(float *data, int size) {

	int gid = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + (threadIdx.y * blockDim.x + threadIdx.x);
    
    if(gid >= size/2)     return;

	if (data[gid] < data[gid + size/2]) {
        data[gid] = data[gid + size/2];
    }

    /*In case the reduction size is odd, there will be a mismatched element. The last thread
      will have to cover it as well */
    if (size % 2 != 0) {
        if (gid == size/2 - 1) {
            if (data[gid] < data[size - 1]) {
                data[gid] = data[size - 1];
            }
        }
    }
}

/**
 * Kernel 4
 * Get the maximum value in data.
 */
__global__ void reduce_max2_kernel(float* data, int size) {
    
    // Shared memory 
    extern __shared__ float tmp[];
    
    int gid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);
    if (gid >= size) return; 

    // Load data in shared memory 
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    tmp[tid] = data[gid];
    
    __syncthreads();

	int mysize = blockDim.x;
	
	if ( (blockIdx.x == gridDim.x-1) && ( (blockDim.x * gridDim.x - size) > 0))
	{
	 	mysize = blockDim.x - (blockDim.x * gridDim.x - size);
	}

    // Reduction in shared memory 
    for (unsigned int s=mysize/2; s>0; s/=2)
	{   
		if (tid < s ) {  
            if (tmp[tid+s] > tmp[tid]) {
				tmp[tid] = tmp[tid+s];  // max(tmp[tid], tmp[tid+s])
			}  			
		}        
		
		// If size is not even 
		if ( (size % 2 != 0) && tid == 0 && tmp[size-1] > tmp[0]  ) {
			tmp[0] = tmp[size-1]; 
		}

		__syncthreads();
    }
    
    /* The thread 0 of each block writes the final result of the reduction
     * in the device's global memory given as a parameter (g_odata[]) */
    if (tid == 0) {
        data[blockIdx.x] = tmp[tid];
    }
}


/**
 * Kernel 5
 * Reduce the heat around the team. 
 */
__global__ void team_actions_kernel(float *surface, int rows, int columns, int paddingColumns, int teamX, int teamY, int squared_radius) {

	int gid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);
		
	if (gid >= rows*paddingColumns) return;

	int i = gid / paddingColumns;
	int j = gid % paddingColumns;

	if (i <= 0 || i >= rows-1 || j <= 0 || j >= columns-1) return; // Out of heated surface 

	if ( (teamX - i)*(teamX - i) + (teamY-j)*(teamY-j) >squared_radius) return ;

	accessMat( surface, i, j) = accessMat( surface, i, j ) * 0.75;
}


/**
 * Kernel 6
 * Update the heat of the surface.
 */
__global__ void update_heat_kernel(float *surface, int rows, int paddingColumns, int x, int y, int heat) {

	int gid = (blockIdx.y * gridDim.x + blockIdx.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x + threadIdx.x);
		
	if (gid >= rows*paddingColumns) return;

	int i = gid / paddingColumns;
	int j = gid % paddingColumns;

	if (i == x || j == y) {
		accessMat(surface, i, j) = heat;
	}
}

 /*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j,t;

	// Simulation data
	int rows, columns, max_iter;
	float *surface, *surfaceCopy;
	int num_teams, num_focal;
	Team *teams;
	FocalPoint *focal;

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc<2) {
		fprintf(stderr,"-- Error in arguments: No arguments\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	int read_from_file = ! strcmp( argv[1], "-f" );
	/* 1.2. Read configuration from file */
	if ( read_from_file ) {
		/* 1.2.1. Open file */
		if (argc<3) {
			fprintf(stderr,"-- Error in arguments: file-name argument missing\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
		FILE *args = cp_abrir_fichero( argv[2] );
		if ( args == NULL ) {
			fprintf(stderr,"-- Error in file: not found: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}	

		/* 1.2.2. Read surface and maximum number of iterations */
		int ok;
		ok = fscanf(args, "%d %d %d", &rows, &columns, &max_iter);
		if ( ok != 3 ) {
			fprintf(stderr,"-- Error in file: reading rows, columns, max_iter from file: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}

		surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
		surfaceCopy = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );

		if ( surface == NULL || surfaceCopy == NULL ) {
			fprintf(stderr,"-- Error allocating: surface structures\n");
			exit( EXIT_FAILURE );
		}

		/* 1.2.3. Teams information */
		ok = fscanf(args, "%d", &num_teams );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error file, reading num_teams from file: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			exit( EXIT_FAILURE );
		}
		for( i=0; i<num_teams; i++ ) {
			ok = fscanf(args, "%d %d %d", &teams[i].x, &teams[i].y, &teams[i].type);
			if ( ok != 3 ) {
				fprintf(stderr,"-- Error in file: reading team %d from file: %s\n", i, argv[1]);
				exit( EXIT_FAILURE );
			}
		}

		/* 1.2.4. Focal points information */
		ok = fscanf(args, "%d", &num_focal );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error in file: reading num_focal from file: %s\n", argv[1]);
			exit( EXIT_FAILURE );
		}
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( focal == NULL ) {
			fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			exit( EXIT_FAILURE );
		}
		for( i=0; i<num_focal; i++ ) {
			ok = fscanf(args, "%d %d %d %d", &focal[i].x, &focal[i].y, &focal[i].start, &focal[i].heat);
			if ( ok != 4 ) {
				fprintf(stderr,"-- Error in file: reading focal point %d from file: %s\n", i, argv[1]);
				exit( EXIT_FAILURE );
			}
			focal[i].active = 0;
		}
	}
	/* 1.3. Read configuration from arguments */
	else {
		/* 1.3.1. Check minimum number of arguments */
		if (argc<6) {
			fprintf(stderr, "-- Error in arguments: not enough arguments when reading configuration from the command line\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}

		/* 1.3.2. Surface and maximum number of iterations */
		rows = atoi( argv[1] );
		columns = atoi( argv[2] );
		max_iter = atoi( argv[3] );

		surface = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );
		surfaceCopy = (float *)malloc( sizeof(float) * (size_t)rows * (size_t)columns );

		/* 1.3.3. Teams information */
		num_teams = atoi( argv[4] );
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			exit( EXIT_FAILURE );
		}
		if ( argc < num_teams*3 + 5 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for %d teams\n", num_teams );
			exit( EXIT_FAILURE );
		}
		for( i=0; i<num_teams; i++ ) {
			teams[i].x = atoi( argv[5+i*3] );
			teams[i].y = atoi( argv[6+i*3] );
			teams[i].type = atoi( argv[7+i*3] );
		}

		/* 1.3.4. Focal points information */
		int focal_args = 5 + i*3;
		if ( argc < focal_args+1 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for the number of focal points\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
		num_focal = atoi( argv[focal_args] );
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			exit( EXIT_FAILURE );
		}
		if ( argc < focal_args + 1 + num_focal*4 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for %d focal points\n", num_focal );
			exit( EXIT_FAILURE );
		}
		for( i=0; i<num_focal; i++ ) {
			focal[i].x = atoi( argv[focal_args+i*4+1] );
			focal[i].y = atoi( argv[focal_args+i*4+2] );
			focal[i].start = atoi( argv[focal_args+i*4+3] );
			focal[i].heat = atoi( argv[focal_args+i*4+4] );
			focal[i].active = 0;
		}

		/* 1.3.5. Sanity check: No extra arguments at the end of line */
		if ( argc > focal_args+i*4+1 ) {
			fprintf(stderr,"-- Error in arguments: extra arguments at the end of the command line\n");
			show_usage( argv[0] );
			exit( EXIT_FAILURE );
		}
	}


#ifdef DEBUG
	/* 1.4. Print arguments */
	printf("Arguments, Rows: %d, Columns: %d, max_iter: %d\n", rows, columns, max_iter);
	printf("Arguments, Teams: %d, Focal points: %d\n", num_teams, num_focal );
	for( i=0; i<num_teams; i++ ) {
		printf("\tTeam %d, position (%d,%d), type: %d\n", i, teams[i].x, teams[i].y, teams[i].type );
	}
	for( i=0; i<num_focal; i++ ) {
		printf("\tFocal_point %d, position (%d,%d), start time: %d, temperature: %d\n", i, 
		focal[i].x,
		focal[i].y,
		focal[i].start,
		focal[i].heat );
	}
#endif // DEBUG

	/* 2. Select GPU and start global timer */
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */
	/* GLOBAL variables */
	int SQUARED_RADIUS_TYPE_1 = RADIUS_TYPE_1 * RADIUS_TYPE_1;
	int SQUARED_RADIUS_TYPE_2_3 = RADIUS_TYPE_2_3 * RADIUS_TYPE_2_3;

	int TRANSACTION_SEGMENT_BYTES = 128;
	int NUM_THREADS_PER_BLOCK = 128;


	/* Make columns a multiple of TRANSACTION_SEGMENT_BYTES */
	int paddingColumns;
	if (columns % (TRANSACTION_SEGMENT_BYTES/sizeof(float)) == 0) {
		paddingColumns = columns;

	} else {
		paddingColumns = (columns / (TRANSACTION_SEGMENT_BYTES/sizeof(float)) + 1 ) * (TRANSACTION_SEGMENT_BYTES/sizeof(float));
		
		surface = (float *) realloc(surface, rows * paddingColumns * sizeof(float));
		surfaceCopy = (float *) realloc(surfaceCopy, rows * paddingColumns * sizeof(float));

		if ( surface == NULL || surfaceCopy == NULL ) {
			fprintf(stderr,"-- Error RE-allocating: surface structures\n");
			exit( EXIT_FAILURE );
		}
	}
	
	

	/* Geometry of grids and blocks */
	int realNumValues = rows * paddingColumns;
	//int nearestUpperPow2 = pow(2,ceil(log2((double) realNumValues)));
	int numValues = realNumValues;

	int numBlocks = numValues / NUM_THREADS_PER_BLOCK;
	if (numValues % NUM_THREADS_PER_BLOCK != 0) {
		numBlocks++;
	}

	dim3 blockSize(NUM_THREADS_PER_BLOCK);
	dim3 gridSize(numBlocks);
	

	/* Allocate surface and surfaceCopy in DEVICE */
	float *devSurface, *devSurfaceCopy, *dev_global_residual;

	cudaMalloc((void **) &devSurface, numValues*sizeof(float));
	CHECK_ERROR("allocating devSurface");

	cudaMalloc((void **) &devSurfaceCopy, numValues*sizeof(float));
	CHECK_ERROR("allocating devSurfaceCopy");

	cudaMalloc((void **) &dev_global_residual, numValues*sizeof(float));
	CHECK_ERROR("allocating dev_global_residual");



    /* Auxiliar variables */
	int fRows = rows - 1;
	int fColumns = columns -1;
	float *temp;

	/* 3. Initialize surface */
	for( i=0; i<rows; i++ )
		for( j=0; j<paddingColumns; j++ )
			accessMat( surface, i, j ) = 0.0;



	/* 4. Simulation */
	int iter;
	int flag_stability = 0;
	int first_activation = 0;
	for( iter=0; iter<max_iter && ! flag_stability; iter++ ) {

		/* 4.1. Activate focal points */
		int num_deactivated = 0;
		for( i=0; i<num_focal; i++ ) {
			if ( focal[i].start == iter ) {
				focal[i].active = 1;
				if ( ! first_activation ) first_activation = 1;
			}
			// Count focal points already deactivated by a team
			if ( focal[i].active == 2 ) num_deactivated++;
		}

		if (!first_activation) continue;


		/* 4.2. Propagate heat (10 steps per each team movement) */
		float global_residual = 0.0f;
		int step;
		bool thresshold_passed = false;
		for( step=0; step<10; step++ )	{
			/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) {
				if ( focal[i].active != 1 ) continue;
				accessMat( surface, focal[i].x, focal[i].y ) = focal[i].heat;
			}

			/* 4.2.2. Copy values of the surface in ancillary structure (Skip borders) */
			temp = surface;
			surface = surfaceCopy;
			surfaceCopy = temp;

			/* Copy surface from HOST to DEVICE */
			cudaMemcpy(devSurfaceCopy, surfaceCopy, realNumValues*sizeof(float), cudaMemcpyHostToDevice);
			CHECK_ERROR("copying surfaceCopy from HOST to DEVICE");


			/* 4.2.3. Update surface values (skip borders) */
			propagate_kernel<<<gridSize, blockSize>>>(devSurface, devSurfaceCopy, rows, columns, paddingColumns);
			CHECK_ERROR("propagate_kernel");
			
			
			/* 4.2.4. Compute the maximum residual difference (absolute value) */
			if (num_deactivated == num_focal && !thresshold_passed) {
				
				/*Compute the residual difference of every position */
				difference_kernel<<<gridSize, blockSize>>>(dev_global_residual, devSurface, devSurfaceCopy, rows, columns, paddingColumns);
				CHECK_ERROR("difference_kernel");
	

				/* Reduction */
				int redSize = numValues;
				int sharedMemorySize = NUM_THREADS_PER_BLOCK * sizeof(float);
				
				while ( redSize > 1 ) 
				{
					int reductionBlocks = redSize / NUM_THREADS_PER_BLOCK;
					if (redSize % NUM_THREADS_PER_BLOCK != 0) {
						reductionBlocks++;
					}
					
					// Make the reduction of the residual difference in the corresponding level 
					reduce_max2_kernel<<< reductionBlocks, blockSize, sharedMemorySize >>>(dev_global_residual, redSize);
					CHECK_ERROR("reduction");

					// Update redSize to the number of blocks of the previous iteration 
					redSize = reductionBlocks;
				} 
				
				
				/* Copy the maximum residual difference from DEVICE to HOST */
				cudaMemcpy(&global_residual, dev_global_residual, sizeof(float), cudaMemcpyDeviceToHost);
				CHECK_ERROR("getting global_residual");

				if (global_residual >= THRESHOLD) {
					thresshold_passed = true;
				}				

			}// end IF 

			/* Copy surface from DEVICE to HOST */
			cudaMemcpy(surface, devSurface,  realNumValues*sizeof(float), cudaMemcpyDeviceToHost);
			CHECK_ERROR("copying surface from DEVICE to HOST");

		} // end 10 steps


		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		if( num_deactivated == num_focal && global_residual < THRESHOLD ) flag_stability = 1;

		if (num_focal != num_deactivated) {
			/* 4.3. Move teams */
			for( t=0; t<num_teams; t++ ) {
				/* 4.3.1. Choose nearest focal point */
				float distance = FLT_MAX;
				int target = -1;
				for( j=0; j<num_focal; j++ ) {
					if ( focal[j].active != 1 ) continue; // Skip non-active focal points
		
					float squared_local_distance = (focal[j].x - teams[t].x)*(focal[j].x - teams[t].x) + (focal[j].y - teams[t].y)*(focal[j].y - teams[t].y);
					if ( squared_local_distance < distance ) {
						distance = squared_local_distance;
						target = j;
					}
				}
				/* 4.3.2. Annotate target for the next stage */
				teams[t].target = target;

				/* 4.3.3. No active focal point to choose, no movement */
				if ( target == -1 ) continue; 

				/* 4.3.4. Move in the focal point direction */
				if ( teams[t].type == 1 ) { 
					// Type 1: Can move in diagonal
					if ( focal[target].x < teams[t].x ) teams[t].x--;
					else if ( focal[target].x > teams[t].x ) teams[t].x++;
					if ( focal[target].y < teams[t].y ) teams[t].y--;
					else if ( focal[target].y > teams[t].y ) teams[t].y++;
				}
				else if ( teams[t].type == 2 ) { 
					// Type 2: First in horizontal direction, then in vertical direction
					if ( focal[target].y < teams[t].y ) teams[t].y--;
					else if ( focal[target].y > teams[t].y ) teams[t].y++;
					else if ( focal[target].x < teams[t].x ) teams[t].x--;
					else if ( focal[target].x > teams[t].x ) teams[t].x++;
				}
				else {
					// Type 3: First in vertical direction, then in horizontal direction
					if ( focal[target].x < teams[t].x ) teams[t].x--;
					else if ( focal[target].x > teams[t].x ) teams[t].x++;
					else if ( focal[target].y < teams[t].y ) teams[t].y--;
					else if ( focal[target].y > teams[t].y ) teams[t].y++;
				}
			} // end team movements
		} // end IF (num_focals != num_deactivated)

		/* 4.4. Team actions */
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y 
				&& focal[target].active == 1 )
				focal[target].active = 2;

			/* 4.4.2. Reduce heat in a circle around the team */
			
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 ) {
				for( i=teams[t].x-RADIUS_TYPE_1; i<=teams[t].x+RADIUS_TYPE_1; i++ ) {
					for( j=teams[t].y-RADIUS_TYPE_1; j<=teams[t].y+RADIUS_TYPE_1; j++ ) {
						if ( i<1 || i>=fRows || j<1 || j>=fColumns ) continue; // Out of the heated surface
					
						float squared_distance = (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j);
						if ( squared_distance <= SQUARED_RADIUS_TYPE_1 ) {
							accessMat( surface, i, j ) = accessMat( surface, i, j ) * 0.75; // Team efficiency factor
						}
					}
				}

			} else {
				for( i=teams[t].x-RADIUS_TYPE_2_3; i<=teams[t].x+RADIUS_TYPE_2_3; i++ ) {
					for( j=teams[t].y-RADIUS_TYPE_2_3; j<=teams[t].y+RADIUS_TYPE_2_3; j++ ) {
						if ( i<1 || i>=fRows || j<1 || j>=fColumns ) continue; // Out of the heated surface
					
						float squared_distance = (teams[t].x - i)*(teams[t].x - i) + (teams[t].y - j)*(teams[t].y - j);
						if ( squared_distance <= SQUARED_RADIUS_TYPE_2_3 ) {
							accessMat( surface, i, j ) = accessMat( surface, i, j ) * 0.75; // Team efficiency factor
						}
					}
				}
			}

			

			//team_actions_kernel<<<gridSize, blockSize>>>(devSurface, rows, columns, paddingColumns, teams[t].x, teams[t].y, squared_radius);
			//CHECK_ERROR("team actions");

		} // end team actions 

		/* Copy surface from DEVICE to host */
		//cudaMemcpy(surface, devSurface, realNumValues*sizeof(float), cudaMemcpyDeviceToHost);
		//CHECK_ERROR("copying surface from DEVICE to host 2");

#ifdef DEBUG
		/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
		print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
#endif // DEBUG
	}
	
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	cudaDeviceSynchronize();
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	//printf("\n");
	/* 6.1. Total computation time */
	printf("\nTime: %lf\n", ttotal );
	/* 6.2. Results: Number of iterations, position of teams, residual heat on the focal points */
	printf("Result: %d", iter);
	/*
	for (i=0; i<num_teams; i++)
		printf(" %d %d", teams[i].x, teams[i].y );
	*/
	for (i=0; i<num_focal; i++)
		printf(" %.6f", accessMat( surface, focal[i].x, focal[i].y ) );
	printf("\n");

	/* 7. Free resources */	
	free( teams );
	free( focal );
	free( surface );
	free( surfaceCopy );

	/* 8. End */
	return 0;
}
