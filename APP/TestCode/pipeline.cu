
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNEL_NUM 3
#define NUM_REPS 5
#define TILE_DIM 32

__global__ void rgb2gray(uint8_t* out_img, uint8_t* in_img,
    int width, int height, int channels);
__global__ void doub_thresh(uint8_t* out_img, uint8_t* in_img,
    uint8_t lower_limit, uint8_t upper_limit,
    int width, int height);
__global__ void thresh2lanes(uint8_t* red_roads_img, uint8_t* edges_img,
                        int width, int height, int channels);
__global__ void copy(uint8_t* out_img, uint8_t* in_img,
    int width, int height, int channels);


int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int width, height, channels;
    uint8_t* rgb_in = stbi_load("image.jpg", &width, &height, &channels, 0);
    size_t rgb_size = width * height * CHANNEL_NUM * sizeof(uint8_t);
    printf("%d %d %d\n", width, height, channels);

    uint8_t * red_roads_out;
    red_roads_out = (uint8_t*)malloc(width * height * CHANNEL_NUM);
    size_t red_roads_size = rgb_size;

    uint8_t* gray_out;
    gray_out = (uint8_t*)malloc(width * height * 1);
    size_t gray_size = width * height * 1 * sizeof(uint8_t);

    uint8_t* edges_out;
    edges_out = (uint8_t*)malloc(width * height * 1);
    size_t edges_size = gray_size;

    uint8_t* d_rgb_in; uint8_t* d_gray_out; uint8_t* d_edges_out; uint8_t* d_red_roads_out;
    cudaMalloc((void**)&d_rgb_in, rgb_size);
    cudaMalloc((void**)&d_gray_out, gray_size);
    cudaMalloc((void**)&d_edges_out, edges_size);
    cudaMalloc((void**)&d_red_roads_out, red_roads_size);

    dim3 dimGrid(ceil(width / 32.0), ceil(height / 32.0), 1);
    dim3 dimBlock(32, 32, 1);

    cudaEventRecord(start);
    for (int rep = 0; rep < NUM_REPS; rep++) {
        cudaMemcpy(d_rgb_in, rgb_in, rgb_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_red_roads_out, d_rgb_in, red_roads_size, cudaMemcpyDeviceToDevice);


        rgb2gray << <dimGrid, dimBlock >> > (d_gray_out, d_rgb_in, width, height, CHANNEL_NUM);

        doub_thresh << <dimGrid, dimBlock >> > (d_edges_out, d_gray_out, 180, 250, width, height);

        thresh2lanes << <dimGrid, dimBlock >> > (d_red_roads_out, d_edges_out, width, height, CHANNEL_NUM);


        // cudaMemcpy(gray_out, d_gray_out, gray_size, cudaMemcpyDeviceToHost);
        // cudaMemcpy(edges_out, d_edges_out, gray_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(red_roads_out, d_red_roads_out, red_roads_size, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("time: %f ms", milliseconds / NUM_REPS);

    // stbi_write_jpg("results/out.jpg", width, height, 1, gray_out, 100);
    // stbi_write_jpg("results/edges.jpg", width, height, 1, edges_out, 100);
    stbi_write_jpg("red_lanes.jpg", width, height, CHANNEL_NUM, red_roads_out, 100);
    
    stbi_image_free(rgb_in);
    free(gray_out);
    free(edges_out);
    free(red_roads_out);

    cudaFree(d_rgb_in);
    cudaFree(d_gray_out);
    cudaFree(d_edges_out);
    cudaFree(d_red_roads_out);

    return 0;
}

__global__ void copy(uint8_t* out_img, uint8_t* in_img,
    int width, int height, int channels) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int base_index = row * width + col;

    if (row < height && col < width) {
        out_img[base_index * channels + 0] = in_img[base_index * channels + 0];
        out_img[base_index * channels + 1] = in_img[base_index * channels + 1];
        out_img[base_index * channels + 2] = in_img[base_index * channels + 2];
    }
}

__global__ void rgb2gray(uint8_t * out_img, uint8_t * in_img, 
                         int width, int height, int channels) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int base_index = row * width + col;

    if (row < height && col < width) {
        uint8_t red = in_img[base_index * channels + 0];
        uint8_t green = in_img[base_index * channels + 1];
        uint8_t blue = in_img[base_index * channels + 2];
        uint8_t gray = 0.21f * red + 0.71f * green + 0.07 * blue;
        // uint8_t gray = (red + green +  blue) / 3;

        out_img[base_index] = gray;
    }
}

__global__ void doub_thresh(uint8_t* out_img, uint8_t* in_img, 
                            uint8_t lower_limit, uint8_t upper_limit, 
                            int width, int height) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int base_index = row * width + col;

    if (row < height && col < width) {
        uint8_t pixel = in_img[base_index];
        if (pixel > lower_limit && pixel < upper_limit)
            out_img[base_index] = 240;
        else out_img[base_index] = 0;
    }
}


__global__ void thresh2lanes(uint8_t* red_roads_img, uint8_t* edges_img,
    int width, int height, int channels) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    int base_index = row * width + col;
    if (row < height && col < width) {

        if (row > (height * 0.6) && edges_img[base_index] != 0) {
            // coalescing may be needed here because edges are vertical
            // but the input image need to be transposed from the beggining (read)

            red_roads_img[base_index * channels + 0] = 250; // red
            red_roads_img[base_index * channels + 1] = 0; // no green
            red_roads_img[base_index * channels + 2] = 0; // no blue
        }
    }
}
