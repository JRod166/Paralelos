#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>


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
    if(rank==0){        
        for(int i=0;i<sizeVector;i++){
            vector[i]=i*1;      
        }       
        for(int i=1;i<size;i++){            
            MPI_Send(vector,sizeVector,MPI_INT,i,10,MPI_COMM_WORLD);                        
        }   
        int sumaTotal=0;
        for(int i=1;i<size;i++){
            MPI_Recv(&suma, 1000, MPI_INT, i, 10, MPI_COMM_WORLD,&status);
            sumaTotal+=suma;
        }
        printf("%d",sumaTotal);

    }   
    if(rank!=0){
        int aux=rank-1;     
        MPI_Recv(vector, sizeVector, MPI_INT, 0, 10, MPI_COMM_WORLD,&status);
        ini=aux*(sizeVector/aux1)+(aux<resto?aux:resto);
        fin=ini+(sizeVector/aux1)+(aux<resto);
        for(int i=ini;i<fin;i++){
            if(i%2==0) suma += vector[i];
            else    suma -= vector[i];
        }
        MPI_Send (&suma, 100, MPI_INT, 0, 10, MPI_COMM_WORLD);  
    }   
    MPI_Finalize();
    free(vector);

}