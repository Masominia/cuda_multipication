#include<cuda.h>
#include<stdio.h>
#include<stdlib.h> 
#include <math.h>
#include <time.h>
#define TILE_DIM 64




//Matrix multiplication kernel without shared memory
__global__ void MatrixMulKernel(int *Md, int *Nd, int *Pd, int Width)
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    int row = blockIdx.y*blockDim.y+threadIdx.y;

    //Pvalue stores the Pd element that is computed by the thread
    int Pvalue = 0;
    if(col<Width && row < Width)
    {
            for(int k = 0; k < Width ; ++k) 
            {
                int Mdelement = Md[row*Width + k];
                int Ndelement = Nd[k*Width + col];
                Pvalue += (Mdelement*Ndelement);
        
            }
            Pd[row*Width + col] = Pvalue;
    }
}

// shared

__global__ void MatrixMultShKernel(int* Md, int* Nd, int* Pd, int Width)
{
    int CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ int Mds[TILE_DIM][TILE_DIM];
    __shared__ int Nds[TILE_DIM][TILE_DIM];

    for (int k = 0; k < ((Width - 1)/TILE_DIM)+1; k++) {

         if (k*TILE_DIM + threadIdx.x < Width && Row < Width)
             Mds[threadIdx.y][threadIdx.x] = Md[Row*Width + k*TILE_DIM + threadIdx.x];
         else
             Mds[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < Width && Col < Width)
             Nds[threadIdx.y][threadIdx.x] = Nd[(k*TILE_DIM + threadIdx.y)*Width + Col];
         else
             Nds[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += Mds[threadIdx.y][n] * Nds[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < Width && Col < Width)
        Pd[((blockIdx.y * blockDim.y + threadIdx.y)*Width) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}


__global__ void MatrixMulShKernel(int *Md, int *Nd, int *Pd, int Width ){
        //Taking shared array to break the MAtrix in Tile widht and fatch them in that array per ele
	__shared__ int Mds [TILE_DIM][TILE_DIM] ;
    __shared__ int Nds [TILE_DIM][TILE_DIM] ;

        // calculate thread id
        int col = blockIdx.x*blockDim.x+threadIdx.x;
        int row = blockIdx.y*blockDim.y+threadIdx.y;

    if(col<Width && row < Width)
    {
    for (int m = 0 ; m<Width/TILE_DIM ; m++ ) // m indicate number of phase
   	{
        Mds[threadIdx.y][threadIdx.x] =  Md[row*Width + (m*TILE_DIM + threadIdx.x)]  ;
        Nds[threadIdx.y][threadIdx.x] =  Nd[ ( m*TILE_DIM + threadIdx.y) * Width + col] ;
        __syncthreads() ; // for syncronizeing the threads

         	// Do for tile
        for ( int k = 0; k<TILE_DIM ; k++ )
            Pd[row*Width + col]+= Mds[threadIdx.x][k] * Nds[k][threadIdx.y] ;
    __syncthreads() ; // for syncronizeing the threads
    }
    }
}


int main(void) {
    
  unsigned  int Width ;
    printf("\n\n Enter Width:");
    scanf("%d",&Width);

//        int M[Width*Width], N[Width*Width], P[Width*Width];

  int * M = (int *) malloc(Width*Width*sizeof(int));
  int * N = (int *) malloc(Width*Width*sizeof(int));
  int * P = (int *) malloc(Width*Width*sizeof(int));
    int *Md, *Nd, *Pd;

    for(int i = 0; i < (Width*Width) ; i++) {
        M[i] = 5;
        N[i] = 5;
        P[i] = 0;
    }





    unsigned long int size = Width*Width*sizeof(int);
  

    //Transfer M and N to device memory
    cudaMalloc((void**)&Md, size);
    cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);

    cudaMalloc((void**)&Nd, size);
    cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);

    //Allocate P on the device
    cudaMalloc((void**)&Pd,size);

    //Setup the execution configuration
    //dim3 dimBlock(Width,Width);
    //dim3 dimGrid(1,1);

    dim3 dimBlock(TILE_DIM,TILE_DIM,1);
    dim3 dimGrid((Width/TILE_DIM)+1,(Width/TILE_DIM)+1,1);



    if (Width*Width > 1024)
    {
        printf("\n\n enter inside if condi\n\n");
        dimGrid.x = (Width-1)/32+1;
            dimGrid.y = (Width-1)/32+1;
    
        dimBlock.x = 32;
            dimBlock.y =32;



    }




    printf(" Width=%d  dimBlock.x = %d   dimBlock.y = %d   dimGrid.x = %d    dimGrid.y = %d \n\n\n\n",Width, dimBlock.x, dimBlock.y ,   dimGrid.x ,    dimGrid.y);
  

    
    float elapsed=0;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

       //Launch the device computation threads!
       MatrixMulKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd,Width);

       //Transfer P from device to host
       cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);
   
       //Free device matrices
       //cudaFree(Md);
       //cudaFree(Nd);
       //cudaFree(Pd);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("The elapsed time in gpu was %.2f ms", elapsed);
    printf("\n");

    float elapsed_sh=0;
    cudaEvent_t start2, stop2;

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2, 0);

       //Launch the device computation threads!
       MatrixMultShKernel<<<dimGrid,dimBlock>>>(Md,Nd,Pd,Width);

       //Transfer P from device to host
       //cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);
   
       //Free device matrices
       cudaFree(Md);
       cudaFree(Nd);
       cudaFree(Pd);


    cudaEventRecord(stop2, 0);
    cudaEventSynchronize (stop2);

    cudaEventElapsedTime(&elapsed_sh, start2, stop2);

    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    printf("The elapsed time in gpu was %.2f ms", elapsed_sh);
    printf("\n");


    float t=clock();
    time_t begin,end;
    time (&begin);
    int *cpu_C;
    cpu_C=new int[Width*Width];

    // Now do the matrix multiplication on the CPU
    int sum;
    for (int row=0; row<Width; row++){
        for (int col=0; col<Width; col++){
            sum = 0.f;
            for (int n=0; n<Width; n++){
                sum += M[row*Width+n]*N[n*Width+col];
            }
            cpu_C[row*Width+col] = sum;
        }
    }
    time (&end);
    t=clock()-t;
    double dif = difftime (end,begin);
    printf ("Elasped time is %.2lf ms.", dif*1000 );
    printf("\n");
    printf ("speed up, cpu/gpu without shared memory = %.2lf", dif*1000/elapsed);
    printf("\n");
    printf ("speed up, cpu/gpu with shared memory = %.2lf", dif*1000/elapsed_sh);
    float es=t/CLOCKS_PER_SEC;
    printf("cpu %.3f",es);
    printf("\n");
    printf ("speed up, cpu/gpu without shared memory = %.2lf", es*1000/elapsed);
    printf("\n");
    printf ("speed up, cpu/gpu with shared memory = %.2lf", es*1000/elapsed_sh);

//Output result matrix

/*     for (int row=0; row<Width*Width; row++){
            
      
            printf("%d   ",P[row]) ;                                                                                                                                                                              
            if((row+1)%Width==0)
            printf("\n");
        }
*/





    int err = 0;
    // Check the result and make sure it is correct

     for (int row=0; row<Width; row++){
        for (int col=0; col<Width; col++){
      
            err += cpu_C[row * Width + col] - P[row * Width + col];                                                                                                                                                                                        
        }
    }

                                                                                                                                                                                                                                                                     
    if(err!=0)
            printf( "\n\nError in matmul\n\n\n");
    else 
        printf("\n\nNo ERROR\n\n\n");

free(M);
free(N);
free(P);

    return 0;


}
