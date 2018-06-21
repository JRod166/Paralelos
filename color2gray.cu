#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>1
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"
#include "GpuTimer.h"

#define NUM_TREADS 1024

using namespace cv;
using namespace std;

// cpu implementation
void rgb2grayCPU(unsigned char* color, unsigned char* gray, int numRows, int numCols, int numChannels) {
    int grayOffset, colorOffset;

    ///recorre secuencialmente todos los pixeles de la imagen

    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            // linearize pixel coordinate tuple (i, j)
            //formula para el offset
            grayOffset = i * numCols + j;
            colorOffset = grayOffset * numChannels;

            //formula para convertir a gris
            gray[grayOffset] = (0.21 * color[colorOffset + 2]) +
                               (0.71 * color[colorOffset + 1]) +
                               (0.07 * color[colorOffset]);
       }
   }
}

// gpu implementation
__global__ void rgb2grayGPU(unsigned char* Pout, unsigned char* Pin, int width, int height, int numChannels) {
    // coordenadas
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    // linearize coordinates for data access
    //formula para el offset tomando en cuenta las coordenadas del thread
    int grayOffset = row * width + col;
    int colorOffset = grayOffset * numChannels;

    ///verifica que el pixel a evaluar exista
    if ((col < width) && (row < height)) {
        ///convierte a gris
        Pout[grayOffset] = (0.21 * Pin[colorOffset + 2]) +
                           (0.71 * Pin[colorOffset + 1]) +
                           (0.07 * Pin[colorOffset]);
    }
}

__global__
void colorToGrayscaleConversion(unsigned char* Pout, unsigned char* Pin, int width, int height, int numChannels){

    //coordenadas
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    int row = threadIdx.y + blockIdx.y*blockDim.y;

    if(col < with && row < height) {
        //formula para el offset teniendo en cuenta las coordenadas
        int greyOffset = row*width + col;
        int rgbOffset = greyOffset* numChannels;

        //consigue valores en rgb
        unsigned char r = Pin [rgbOffset  ];
        unsigned char g = Pin [rgbOffset+1];
        unsigned char b = Pin [rgbOffset+2];

        //convierte a gris
        Pout[grayOffset] = 0.21f*r +0.71f*g +0.07f*b;
    }
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("[!] Filename expected.\n");
        return 0;
    }

    // read image
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        printf("Cannot read image file %s", argv[1]);
        exit(1);
    }

    // parametros
    int imageChannels = 3; //rgb
    int imageWidth = image.cols;
    int imageHeight = image.rows;
    size_t size_rgb = sizeof(unsigned char)*imageWidth*imageHeight*imageChannels;
    size_t size_gray = sizeof(unsigned char)*imageWidth*imageHeight;

    // reserva memoria para imagenes en host
    unsigned char* h_grayImage = (unsigned char*)malloc(size_rgb);
    unsigned char* h_grayImage_CPU = (unsigned char*)malloc(size_rgb);

    // puntero a la imagen rgb en host
    unsigned char* h_rgbImage = image.data;

    // reserva memoria para imagenes en device
    unsigned char* d_rgbImage;
    unsigned char* d_grayImage;
    
    cudaMalloc((void**)&d_rgbImage, size_rgb);
    cudaMalloc((void**)&d_grayImage, size_gray);

    // copia la imagen rgb de host a device
    cudaMemcpy(d_rgbImage, h_rgbImage, size_rgb, cudaMemcpyHostToDevice);

    // parametros de ejecucion
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(ceil(imageWidth/16.0), ceil(imageHeight/16.0), 1);
    //dim3 dimBlock(NUM_THREADS, NUM_THREADS, 1);
    //dim3 dimGrid(ceil(imageWidth/NUM_THREADS), ceil(imageHeight/NUM_THREADS), 1);

    //ejecucion
    //rgb2grayGPU<<<dimGrid, dimBlock>>>(d_grayImage, d_rgbImage, imageWidth, imageHeight, imageChannels);
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(h_rgbImage, h_grayImage_CPU, imageHeight, imageWidth, imageChannels);

    // copia la imagen en gris del device al host
    cudaMemcpy(h_grayImage, d_grayImage, size_gray, cudaMemcpyDeviceToHost);


    // display images
    Mat Image1(imageHeight, imageWidth, CV_8UC1, h_grayImage);
    Mat Image2(imageHeight, imageWidth, CV_8UC1, h_grayImage_CPU);
    namedWindow("CPUImage", WINDOW_NORMAL);
    namedWindow("GPUImage", WINDOW_NORMAL);
    imshow("GPUImage",Image1);
    imshow("CPUImage",Image2);
    waitKey(0);

    // libera espacios de memoria
    image.release();
    Image1.release();
    Image2.release();
    free(h_grayImage);
    free(h_grayImage_CPU);
    cudaFree(d_rgbImage); cudaFree(d_grayImage);

    return 0;
}