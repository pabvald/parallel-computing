/* Grupo 105 
 * Pablo Valdunciel Sánchez 
 * Iván González Rincón 
 */

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
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<float.h>
#include<mpi.h>
#include<cputils.h>

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

#ifdef ERRORCHECK
	void errorCheck( int, char* );
	void printMatrix( float*, int, int);
#endif


/* Macro function to simplify accessing with two coordinates to a flattened array */
#define accessMat( arr, exp1, exp2 )	arr[ (exp1) * columns + (exp2) ]


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



/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	int i,j,t;

	// Simulation data
	int rows, columns, max_iter;
	int num_teams, num_focal;
	Team *teams;
	FocalPoint *focal;

	/* 0. Initialize MPI */
	int rank;
	int nprocs;
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

	/* 1. Read simulation arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc<2) {
		fprintf(stderr,"-- Error in arguments: No arguments\n");
		show_usage( argv[0] );
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

	int read_from_file = ! strcmp( argv[1], "-f" );
	/* 1.2. Read configuration from file */
	if ( read_from_file ) {
		/* 1.2.1. Open file */
		if (argc<3) {
			fprintf(stderr,"-- Error in arguments: file-name argument missing\n");
			show_usage( argv[0] );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		FILE *args = cp_abrir_fichero( argv[2] );
		if ( args == NULL ) {
			fprintf(stderr,"-- Error in file: not found: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}	

		/* 1.2.2. Read surface size and maximum number of iterations */
		int ok;
		ok = fscanf(args, "%d %d %d", &rows, &columns, &max_iter);
		if ( ok != 3 ) {
			fprintf(stderr,"-- Error in file: reading rows, columns, max_iter from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}

		/* 1.2.3. Teams information */
		ok = fscanf(args, "%d", &num_teams );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error file, reading num_teams from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		for( i=0; i<num_teams; i++ ) {
			ok = fscanf(args, "%d %d %d", &teams[i].x, &teams[i].y, &teams[i].type);
			if ( ok != 3 ) {
				fprintf(stderr,"-- Error in file: reading team %d from file: %s\n", i, argv[1]);
				MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
			}
		}

		/* 1.2.4. Focal points information */
		ok = fscanf(args, "%d", &num_focal );
		if ( ok != 1 ) {
			fprintf(stderr,"-- Error in file: reading num_focal from file: %s\n", argv[1]);
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( focal == NULL ) {
			fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		for( i=0; i<num_focal; i++ ) {
			ok = fscanf(args, "%d %d %d %d", &focal[i].x, &focal[i].y, &focal[i].start, &focal[i].heat);
			if ( ok != 4 ) {
				fprintf(stderr,"-- Error in file: reading focal point %d from file: %s\n", i, argv[1]);
				MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
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
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}

		/* 1.3.2. Surface size and maximum number of iterations */
		rows = atoi( argv[1] );
		columns = atoi( argv[2] );
		max_iter = atoi( argv[3] );

		/* 1.3.3. Teams information */
		num_teams = atoi( argv[4] );
		teams = (Team *)malloc( sizeof(Team) * (size_t)num_teams );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d teams\n", num_teams );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		if ( argc < num_teams*3 + 5 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for %d teams\n", num_teams );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
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
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		num_focal = atoi( argv[focal_args] );
		focal = (FocalPoint *)malloc( sizeof(FocalPoint) * (size_t)num_focal );
		if ( teams == NULL ) {
			fprintf(stderr,"-- Error allocating: %d focal points\n", num_focal );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		if ( argc < focal_args + 1 + num_focal*4 ) {
			fprintf(stderr,"-- Error in arguments: not enough arguments for %d focal points\n", num_focal );
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
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
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
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

	/* 2. Start global timer */
	MPI_Barrier(MPI_COMM_WORLD);
	double ttotal = cp_Wtime();


	float *residualHeat = (float*)malloc( sizeof(float) * (size_t)num_focal );	
	if ( residualHeat == NULL ) {
		fprintf(stderr,"-- Error allocating: residualHeat\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}	

	/*
	*
	* START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
	*
	*/

	#ifdef ERRORCHECK 
		MPI_Barrier(MPI_COMM_WORLD);
		double timeInitialization = cp_Wtime();
		double timeComm;
		double timeRest;
	#endif

	/* Constants */
	const int RADIUS_TYPE_1_SQUARED = RADIUS_TYPE_1 * RADIUS_TYPE_1;		// Squared RADIUS_TYPE_1
	const int RADIUS_TYPE_2_3_SQUARED = RADIUS_TYPE_2_3 * RADIUS_TYPE_2_3;	// Squared RADIUS_TYPE_2_3

	const int root = 0;			// Root processor
	const int last = nprocs-1;	// Last processor


	/* MPI error handling */ 
	int err;
	int resultlen;  
	char errStr[MPI_MAX_ERROR_STRING];
	MPI_Status stat;

	MPI_Comm_set_errhandler( MPI_COMM_WORLD, MPI_ERRORS_RETURN ); 		//Change error handeler to MPI_ERRORRS_RETURN


	/* Communications - Send & Recieve */
	MPI_Request sendUp;
	MPI_Request sendDown;
	MPI_Status stat1;
	MPI_Status stat2;
	int tagSendBackwards = 1;		// Tag - The communication (send or recv) consists in moving a row from processor n to processor n-1
	int tagSendForwards = 2;		// Tag - The communication (send or rect) consists in moving a row from processor n to processor n+1


	/* Variables neeeded for surface division */	
	int rest;													 // Rest of the surface division. It is equal to 0 if 'rows' is a multiple of 'nprocs'
	int my_begin;												 // Row where the rows owned by this processor start.
	int my_end;													 // Row where the rows owned by this processor end.
	int bandSize;												 // Number of rows in the band ( halos included )
	int displBand;												 // Row where the rows owned by the processor start, after the halo

	int *sizes = (int *) malloc( sizeof(int) * (size_t)nprocs);	 // Number of rows owned by each processor ( halos NOT included )	
	if( sizes == NULL ) {
		fprintf(stderr,"-- Error allocating: sizes\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	}

	float *root_residualHeat;														// Array that will store the residual heat of the focals
	if (rank == root) {
		root_residualHeat = (float*) malloc( sizeof(float) * (size_t)num_focal );	//Only root allocates memory
		if ( root_residualHeat == NULL ) {
			fprintf(stderr,"-- Error allocating: root_residualHeat\n");
			MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
		}
		for( i=0; i<num_focal; i++) root_residualHeat[i] = 0.0;
	}

	// Set rest 
	rest = rows % nprocs;

	// Set processors' sizes 
	for( i=0; i<nprocs; i++) {
		sizes[i] = rows / nprocs;
		if( i < rest) sizes[i]+=1;
	}

	// Set my_begin and my_end
	my_begin = 0;
	for( i=0; i<rank; i++) my_begin += sizes[i]; 
	my_end = my_begin + sizes[rank] - 1;	

	// Set bands' sizes and displacements 	
	if ( rank != 0 && rank != last ) {				
		bandSize = sizes[rank] + 2;	
		displBand = 1;	

	} else if ( rank == 0) {
		bandSize = sizes[rank] + 1;
		displBand = 0;

	} else { // rank == last 			
		bandSize = sizes[rank] + 1;
		displBand = 1;			
	}         


	float *band, *bandCopy, *temp;

	band = (float *) malloc( sizeof( float ) * (size_t)( bandSize ) * (size_t)columns);
	if ( band == NULL ) {
		fprintf(stderr,"-- Error allocating: band structures\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	} 

	bandCopy = (float *) malloc( sizeof( float ) * (size_t)( bandSize ) * (size_t)columns);
	if ( band == NULL ) {
		fprintf(stderr,"-- Error allocating: bandCopy structures\n");
		MPI_Abort( MPI_COMM_WORLD,  EXIT_FAILURE );
	} 
	
		
	/* Skip-borders variables */
	const int fCols = columns -1;
	const int fRows = rows -1;						// Modificación: añadidas las variables fRows y fCols
	const int fBandSize = bandSize - 1;				// bandSize -1  - it is used in 'for' loops to skip border in bands
	  
	
	/* 3.Initialize surface band and surfaceCopy band */
	for ( i=0; i<bandSize; i++) {
		for ( j=0; j<columns; j++) {
			accessMat( band, i, j ) = 0.0f;
			accessMat( bandCopy, i, j ) = 0.0f;
		}
	}

	#ifdef ERRORCHECK 
		MPI_Barrier(MPI_COMM_WORLD);
		timeInitialization = timeInitialization - cp_Wtime();
	#endif 	


	/* 4. Simulation */
	int iter;
	int flag_stability = 0;
	int first_activation = 0;
	for (iter = 0; iter<max_iter && ! flag_stability; iter++ ) {

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
		
		if ( !first_activation ) continue;	// Modificación

		#ifdef ERRORCHECK 
				MPI_Barrier(MPI_COMM_WORLD); 
				timeComm = cp_Wtime();
		#endif

		/* 4.2. Propagate heat (10 steps per each team movement) */
		float my_global_residual = 0.0f;
		float global_residual = 0.0f;
		int step; 
		int thresshold_passed = 0;
		for( step=0; step<10; step++ )	{

			/* 4.2.1. Update heat on active focal points */
			for( i=0; i<num_focal; i++ ) { 
				if ( focal[i].active != 1  || focal[i].x < my_begin || focal[i].x > my_end ) continue;

				accessMat( band, ( focal[i].x - my_begin + displBand ), focal[i].y ) = focal[i].heat;								
			}				
			

			/* Communications - Send and receive halos */
			if ( rank != last) {                            
				// 1.1 Send penultimate row to rank+1
				MPI_Isend( band + ((bandSize-2)*columns), columns, MPI_FLOAT, rank+1, tagSendForwards , MPI_COMM_WORLD, &sendDown);
					
				// 1.2 Receive last row from rank+1
				err = MPI_Recv( band + ((bandSize-1)*columns), columns, MPI_FLOAT, rank+1, tagSendBackwards, MPI_COMM_WORLD, &stat2);				
				#ifdef ERRORCHECK 
					if( err != MPI_SUCCESS ) errorCheck( err, "1.2.");  
				#endif

				err = MPI_Wait( &sendDown, &stat1);
				#ifdef ERRORCHECK  
					if( err != MPI_SUCCESS ) errorCheck( err, "1.1.");
				#endif
			} 
			
			if ( rank != 0 ) {     
				// 2.2. Send second row to rank-1
				MPI_Isend( band + 1*columns, columns, MPI_FLOAT, rank-1, tagSendBackwards, MPI_COMM_WORLD, &sendUp );
				
				// 2.1. Receive first row from rank-1
				err = MPI_Recv( band, columns, MPI_FLOAT, rank-1, tagSendForwards , MPI_COMM_WORLD, &stat2 );
				#ifdef ERRORCHECK 
					if( err != MPI_SUCCESS ) errorCheck( err, "2.1");
				#endif	

				err = MPI_Wait( &sendUp, &stat1);
				#ifdef ERRORCHECK 
					if( err != MPI_SUCCESS ) errorCheck( err, "2.2");
				#endif			
			}		
			
			/* 4.2.2. Copy band to bandCopy, border included */			
			temp = band;
			band = bandCopy;
			bandCopy = temp;


			/* 4.2.3. Update surface values (skip borders) */
			for ( i=1; i<fBandSize; i++) 
				for ( j=1; j<fCols; j++) 
					accessMat( band, i, j ) = (
							accessMat( bandCopy, i+1, j ) +
							accessMat( bandCopy, i-1, j ) +
							accessMat( bandCopy, i, j+1 ) +
							accessMat( bandCopy, i, j-1 ) ) / 4;
			 

			/* 4.2.4. Compute the maximum residual difference (absolute value) */
			if ( num_deactivated == num_focal && !thresshold_passed ) {			
				for ( i=1; i<fBandSize; i++) {
					for ( j=1; j<fCols; j++) {
						if ( fabs( accessMat( band, i, j ) - accessMat( bandCopy, i, j ) ) > my_global_residual ) {			
							my_global_residual = fabs( accessMat( band, i, j ) - accessMat( bandCopy, i, j ) );
							
							if ( my_global_residual >= THRESHOLD ) {
								thresshold_passed = 1; 
								break;
							}
						}
					}
				}
			}
		}  
		
		if ( num_deactivated == num_focal ) {
			/* Each processor gets the maximum of global_residual */
			err = MPI_Allreduce( &my_global_residual, &global_residual, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
			#ifdef ERRORCHECK
				if( err != MPI_SUCCESS ) {
						MPI_Error_string( err, errStr, &resultlen );
						printf("--Error: %s in reduce of global_residual\n\n", errStr);
						MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE );
				}
			#endif 
		}

		#ifdef ERRORCHECK 
				MPI_Barrier(MPI_COMM_WORLD);
				timeComm = cp_Wtime() - timeComm;
				timeRest = cp_Wtime();
		#endif


		/* If the global residual is lower than THRESHOLD, we have reached enough stability, stop simulation at the end of this iteration */
		if( num_deactivated == num_focal && global_residual < THRESHOLD )  flag_stability = 1;
		

		/* 4.3. Move teams */
		if ( num_deactivated != num_focal ) {
			for( t=0; t<num_teams; t++ ) {

				/* 4.3.1. Choose nearest focal point */
				float distance = FLT_MAX;
				int target = -1;
				for( j=0; j<num_focal; j++ ) {
					if ( focal[j].active != 1 ) continue; // Skip non-active focal points
					//float dx = focal[j].x - teams[t].x;					//Modificación: eliminación de variables.
					//float dy = focal[j].y - teams[t].y;
					//float local_distance = sqrtf( dx*dx + dy*dy );		//Modificación: eliminación de las raices cuadradas
					float local_distance =  (focal[j].x - teams[t].x)*(focal[j].x - teams[t].x)  +  (focal[j].y - teams[t].y)*(focal[j].y - teams[t].y);	
					if ( local_distance < distance ) {
						distance = local_distance;
						target = j;
					}
				}

				/* 4.3.2. Annotate target for the next stage */
				teams[t].target = target;

				/* 4.3.3. No active focal point to choose, no movement */
				if ( target == -1 ) continue; 

				/* 4.3.4. Move in the focal point direction */
				if ( teams[t].type == 1) { 
					
					// Type 1: Can move in diagonal
					if ( focal[target].x < teams[t].x ) teams[t].x--;
					else if ( focal[target].x > teams[t].x ) teams[t].x++; // Moficiación: 'else if' en lugar de 'if'
					if ( focal[target].y < teams[t].y ) teams[t].y--;
					else if ( focal[target].y > teams[t].y ) teams[t].y++; // Moficiación: 'else if' en lugar de 'if'
				
				} else if ( teams[t].type == 2) {
					// Type 2: First in horizontal direction, then in vertical direction
					if ( focal[target].y < teams[t].y ) teams[t].y--;
					else if ( focal[target].y > teams[t].y ) teams[t].y++; 
					else if ( focal[target].x < teams[t].x ) teams[t].x--;
					else if ( focal[target].x > teams[t].x ) teams[t].x++;
				
				} else {
					 
					// Type 3: First in vertical direction, then in horizontal direction
					if ( focal[target].x < teams[t].x ) teams[t].x--;
					else if ( focal[target].x > teams[t].x ) teams[t].x++;
					else if ( focal[target].y < teams[t].y ) teams[t].y--;
					else if ( focal[target].y > teams[t].y ) teams[t].y++;
					
				}
			}
		}
		

		/* 4.4. Team actions */
		for( t=0; t<num_teams; t++ ) {
			/* 4.4.1. Deactivate the target focal point when it is reached */
			int target = teams[t].target;
			if ( target != -1 && focal[target].x == teams[t].x && focal[target].y == teams[t].y && focal[target].active == 1 ) 
				focal[target].active = 2;

			/* 4.4.2. Reduce heat in a circle around the team */
			// Influence area of fixed radius depending on type
			if ( teams[t].type == 1 ) {

				for( i=teams[t].x-RADIUS_TYPE_1; i<=teams[t].x+RADIUS_TYPE_1; i++ ) {
					for( j=teams[t].y-RADIUS_TYPE_1; j<=teams[t].y+RADIUS_TYPE_1; j++ ) {
						if ( i < my_begin  ||  i > my_end || i<1 || i>=fRows|| j<1 || j>=fCols ) continue; // Out of the heated surface
						//float dx = teams[t].x - i;
						//float dy = teams[t].y - j;
						//float distance = sqrtf( dx*dx + dy*dy );	//Modificacion 
						if ( (teams[t].x - i)*(teams[t].x - i)  +  (teams[t].y - j)*(teams[t].y - j) <= RADIUS_TYPE_1_SQUARED ) {
							accessMat( band, ( i - my_begin + displBand ), j ) = accessMat( band, ( i - my_begin + displBand ), j )*(0.75);  // Team efficiency factor																									
						}
					}
				}
			} else {
				
				for( i=teams[t].x-RADIUS_TYPE_2_3; i<=teams[t].x+RADIUS_TYPE_2_3; i++ ) {
					for( j=teams[t].y-RADIUS_TYPE_2_3; j<=teams[t].y+RADIUS_TYPE_2_3; j++ ) {
						if (  i < my_begin  ||  i > my_end || i<1 || i>=fRows|| j<1 || j>=fCols ) continue; // Out of the heated surface					

						if ( (teams[t].x - i)*(teams[t].x - i)  +  (teams[t].y - j)*(teams[t].y - j) <= RADIUS_TYPE_2_3_SQUARED ) {
							accessMat( band, ( i - my_begin + displBand ), j ) = accessMat( band, ( i - my_begin + displBand ), j )*(0.75);	// Team efficiency factor																							 // Team efficiency factor
						}
					}
				}
			}
						
		}

		#ifdef DEBUG
				/* 4.5. DEBUG: Print the current state of the simulation at the end of each iteration */
				print_status( iter, rows, columns, surface, num_teams, teams, num_focal, focal, global_residual );
		#endif // DEBUG


		#ifdef ERRORCHECK 
			MPI_Barrier(MPI_COMM_WORLD);
			timeRest =  cp_Wtime() - timeRest;
		#endif
	}
		

	/* Each processor stores the  residual heat of the focals which are in its bands */
	for( i=0; i<num_focal; i++) residualHeat[i] = 0.0;

	for (i=0; i<num_focal; i++) {
		if( focal[i].x >= my_begin &&  focal[i].x <= my_end ) {
			residualHeat[i] = accessMat( band, ( focal[i].x - my_begin + displBand ), focal[i].y );
		}
	} 

	/* The root gets the residual heat of all focals in the surface */
	err = MPI_Reduce(residualHeat, root_residualHeat, num_focal, MPI_FLOAT, MPI_MAX, root, MPI_COMM_WORLD );
	if( err != MPI_SUCCESS ) {
		MPI_Error_string( err, errStr, &resultlen );
		printf("--Error: %s in reduce of residual_heat\n\n", errStr);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE );
	}  

	free( sizes );
	free( band );
	free( bandCopy );
	free( residualHeat );

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 5. Stop global time */
	MPI_Barrier(MPI_COMM_WORLD);
	ttotal = cp_Wtime() - ttotal;

	/* 6. Output for leaderboard */
	if ( rank == root ) {
		printf("\n");
		/* 6.1. Total computation time */
		printf("Time: %lf\n", ttotal );
		/* 6.2. Results: Number of iterations, residual heat on the focal points */
		printf("Result: %d", iter);
		for (i=0; i<num_focal; i++)
			printf(" %.6f", root_residualHeat[i] );
		printf("\n");
	}

	/* 7. Free resources */	
	free( teams );
	free( focal );
	if( rank == root) free( root_residualHeat );
	
	/* 8. End */
	MPI_Finalize();
	return 0;
}

#ifdef ERRORCHECK 

	/*
	* It prints an error.
	*/
	void errorCheck( int errorCode, char *section ) {
		char errStr[MPI_MAX_ERROR_STRING];
		int resultlen;    

		if( errorCode != MPI_SUCCESS ) {
				MPI_Error_string( errorCode, errStr, &resultlen );
				printf("\n--Error: %s  en %s\n\n", errStr, section);
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE );
		}
	}

	/*
	* It prints a matrix.
	*/
	void printMatrix( float *m, int rows, int  columns) {
		int i, j;
		for( i=0; i<rows; i++ ) {
			for( j=0; j<columns; j++ ) {
				printf(" %.2f", accessMat( m, i, j ) );
			}
			printf("\n");
		}
	}
	

#endif
