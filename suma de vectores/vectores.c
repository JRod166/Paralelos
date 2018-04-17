#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

/// mpirun -np 3 ./vectores 11

MPI_Status status;
int suma=0;
int sizeVector=0;
int* vector=NULL;


int main(int argc, char *argv[]){
    int rank,size;  
    int ini,fin;

    sscanf (argv[1],"%i",&sizeVector);

    MPI_Init( &argc, &argv );    
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    int aux1=size-1;
    int resto=sizeVector%aux1;  
    vector=(int*)malloc(sizeVector*sizeof(int));
    //generando vector
    if(rank==0){        
        for(int i=0;i<sizeVector;i++){
            vector[i]=i*1;      
        }       
        //enviando vectores
        for(int i=1;i<size;i++){
            MPI_Send(vector,sizeVector,MPI_INT,i,10,MPI_COMM_WORLD);                        
        }   
        int sumaTotal=0;
        //sumando vectores
        for(int i=1;i<size;i++){
            printf("Proceso: %d ",rank);
            MPI_Recv(&suma, 1000, MPI_INT, i, 10, MPI_COMM_WORLD,&status);
            printf("%d + %d\n",sumaTotal,suma);
            sumaTotal=sumaTotal+suma;
        }
        printf("%d \n",sumaTotal);

    }   
    if(rank!=0){
        int aux=rank-1;     
        MPI_Recv(vector, sizeVector, MPI_INT, 0, 10, MPI_COMM_WORLD,&status);
        ini=aux*(sizeVector/aux1)+(aux<resto?aux:resto);
        fin=ini+(sizeVector/aux1)+(aux<resto);
        printf("Proceso: %d \n",rank);
        for(int i=ini;i<fin;i++){
            printf("%d + %d  \n",suma,vector[i]);
            suma +=vector[i];
        }
        MPI_Send (&suma, 100, MPI_INT, 0, 10, MPI_COMM_WORLD);  
    }   
    free(vector);
    MPI_Finalize();


}