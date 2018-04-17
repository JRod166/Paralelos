#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

/*mpicc -g -Wall -o matrizvector matrizvector.c    mpirun -np 2 ./matrizvector Dos procesos si divides 10 en 5*/
void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int local_n){
  int local_i;
  for(local_i=0;local_i < local_n; local_i++){
    local_z[local_i] = local_x[local_i] + local_y[local_i];
  }
}

void Read_vector(double local_a[], int local_n, int n, char vec_name[], int my_rank, MPI_Comm comm){
  double* a = NULL;
  int i;

  if(my_rank == 0){
    a = malloc(n*sizeof(double));
    printf("ENter the vector %s\n", vec_name);
    for(i = 0; i < n; i++)
      {scanf("%lf", &a[i]);}
     MPI_Scatter(a, local_n, MPI_DOUBLE, local_a, local_n, MPI_DOUBLE,0,comm);
     free(a);
    
  }
    else{
      MPI_Scatter(a,local_n,MPI_DOUBLE,local_a,local_n,MPI_DOUBLE,0,comm);
    }
  

}

void Print_vector(double local_b[], int local_n, int n, char title[], int my_rank, MPI_Comm comm){
  double* b = NULL;
  int i;
  if(my_rank == 0){
    b = malloc(n*sizeof(double));
    MPI_Gather(local_b, local_n, MPI_DOUBLE,b,local_n, MPI_DOUBLE,0,comm);
    printf("%s\n", title);
    for(i = 0; i < n; i++){
      printf("%f ", b[i]);
    }
    printf("\n");
    free(b);
  }
  else{
    MPI_Gather(local_b, local_n, MPI_DOUBLE, b, local_n, MPI_DOUBLE, 0, comm);
  }
}


int main(){
  int comm_sz;
  int my_rank;
  int local_n=5, n=10; 
  double local_a[local_n], local_b[local_n], local_z[local_n];
  char vec_name[10];
  MPI_Init(NULL,NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  Read_vector(local_a,local_n,n,"Vector Sumando 1",my_rank,MPI_COMM_WORLD);
  Read_vector(local_b,local_n,n,"Vector Sumando 2",my_rank,MPI_COMM_WORLD);
  Parallel_vector_sum(local_a,local_b,local_z,local_n);
  Print_vector(local_z,local_n,n,"Vector Contenedor", my_rank,MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}