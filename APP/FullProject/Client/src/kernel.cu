#include "kernel.cuh"

int width = WIDTH * FACTOR, height = HEIGHT * FACTOR, channels = CHANNELS;

dim3 dimGrid(ceil(width / 32.0), ceil(height / 32.0), 1);
dim3 dimBlock(32, 32, 1);

__global__ void rgb2gray(uint8_t* out_img, uint8_t* in_img,
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


/*=============== Qt Versions =================*/



QString rgb_2_gray(uint8_t* d_gray_out, uint8_t* d_rgb_in) {
    rgb2gray << <dimGrid, dimBlock >> > (d_gray_out, d_rgb_in, width, height, channels);
    return "Done";
}

QString doub_threshold(uint8_t* d_edges_out, uint8_t* d_gray_out,
    uint8_t lower_limit, uint8_t upper_limit) {
    doub_thresh << <dimGrid, dimBlock >> > (d_edges_out, d_gray_out, 175, 250, width, height);
    return "Done";
}

QString thresh_2_lanes(uint8_t* d_red_roads_out, uint8_t* d_edges_out) {
    thresh2lanes << <dimGrid, dimBlock >> > (d_red_roads_out, d_edges_out, width, height, channels);
    return "Done";
}