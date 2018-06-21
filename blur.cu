#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "GpuTimer.h"
#include <time.h>

using namespace cv;
using namespace std;

#define FILTER_SIZE 11
#define BLOCK_SIZE 16

__global__ void imgBlurGPU(unsigned char* outImg, unsigned char* inImg, int width, int height) {
    int filterRow, filterCol;
    int cornerRow, cornerCol;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int filterSize = 2*FILTER_SIZE + 1;

    //Indices de filas y columnas
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    //If para que verifique que esta dentro de los limites
    if ((row < height) && (col < width)) {

        int numPixels = 0;
        int cumSum = 0;

        //esquina superior izquierda para hacer blur
        cornerRow = row - FILTER_SIZE;
        cornerCol = col - FILTER_SIZE;


        //Acumula los valores de los pixeles en el filter size
        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                filterRow = cornerRow + i;
                filterCol = cornerCol + j;
                if ((filterRow >= 0) && (filterRow <= height) && (filterCol >= 0) && (filterCol <= width)) {
                    cumSum += inImg[filterRow*width + filterCol];
                    //actualiza la cantidad de pixeles que esta sumando
                    numPixels++;
                }
            }
        }
        //el output es el promedio de los valores en el filtersize (cumSum/numPixels)
        outImg[row*width + col] = (unsigned char)(cumSum / numPixels);
    }
}

__global__
void blurKernel(unsingned char* in, unsigned char* out, int w, int h){
	//Indices de filas y columnas
    int Col = blockIdx.x * blockDim.x + ThreadIdx.x;
    int Row = blockIdx.y * blockDim.y + ThreadIdx.y;

    //If para que verifique que esta dentro de los limites
    if (Col < w && Row < h){
        int pixVal = 0;	//acumula los valores de los pixeles dentro del espacio a hacer blur
        int pixels = 0;	//contador de pixeles sumados
        for(int blurRow= -BLUR_SIZE; blurRow < BLUR_SIZE +1; ++blurRow)
            for(int blurCol= -BLUR_SIZE; blurCol < BLUR_SIZE +1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                ///if para que verifique que el pixel al que intenta acceder existe
                if (curRow > -1 && curRow < h && curCol >-1 && curCol < w){
                    pixVal += in[curRow*w + curCol];
                    pixels ++;
                }
            }
    }
    //el output es el promedio de los valores de los pixeles evaluados
    out[ Row*w +Col ] = (unsigned char) (pixelVal/ pixels);
}

int main(int argc, char *argv[]) {
   
    if (argc == 1) {
        printf("[!] Filename expected.\n");
        return 0;
    }

    // read image
    Mat img;
    img = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if (img.empty()) {
        printf("Cannot read image file %s", argv[1]);
        exit(1);
    }

    // parametros de la imagen
    int imgWidth = img.cols;
    int imgHeight = img.rows;
    size_t imgSize = sizeof(unsigned char)*imgWidth*imgHeight;
    GpuTimer timer;

    // reserva memoria para las imagenes en host
    unsigned char* h_outImg = (unsigned char*)malloc(imgSize);
    unsigned char* h_outImg_CPU = (unsigned char*)malloc(imgSize);

    // puntero a la imagen de entrada
    unsigned char* h_inImg = img.data;

    // reserva memoria para las imagenes en device
    unsigned char* d_inImg;
    unsigned char* d_outImg;

    cudaMalloc((void**)&d_inImg, imgSize);
    cudaMalloc((void**)&d_outImg, imgSize);

    // copia la imagen de host a device
    cudaMemcpy(d_inImg, h_inImg, imgSize, cudaMemcpyHostToDevice);

    // parametros para la ejecucion
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(imgWidth/16.0), ceil(imgHeight/16.0), 1);

    //imgBlurGPU<<<dimGrid, dimBlock>>>(d_outImg, d_inImg, imgWidth, imgHeight);
    blurKernel<<<dimGrid, dimBlock>>>(d_outImg, d_inImg, imgWidth, imgHeight);

    // copia la imagen de salida del device al host
    cudaMemcpy(h_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost);

    // muestra imagenes
    Mat img1(imgHeight, imgWidth, CV_8UC1, h_outImg);
    Mat img2(imgHeight, imgWidth, CV_8UC1, h_outImg_CPU);
    namedWindow("Before", WINDOW_NORMAL);
    imshow("Before", img);
    namedWindow("After (GPU)", WINDOW_NORMAL);
    imshow("After (GPU)", img1);
    namedWindow("After (CPU)", WINDOW_NORMAL);
    imshow("After (CPU)", img2);
    waitKey(0);

    // libera los espacios de memoria
    img.release(); img1.release(); img2.release();
    free(h_outImg_CPU); free(h_outImg);
    cudaFree(d_outImg); cudaFree(d_inImg);

    return 0;
} 