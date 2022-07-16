
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <opencv2/opencv.hpp>

#define CHANNEL_NUM 3


void rgb2gray(uint8_t* input_img, uint8_t* gray_img, int width, int height, int channels);
void threshold(uint8_t* gray_img, uint8_t* threshold_img,
    int width, int height, uint8_t lower_limit, uint8_t upper_limit);
void thresh2lanes(uint8_t* input_img, uint8_t* out_img, uint8_t* thresh, int width, int height, int channels);

using namespace cv;

int main()
{
    int width = 1280, height = 720, channels = 3;
    size_t rgb_size = width * height * channels * sizeof(uint8_t);

    uint8_t* gray_out;
    gray_out = (uint8_t*)malloc(width * height * 1);
    size_t gray_size = width * height * 1 * sizeof(uint8_t);

    uint8_t* edges_out;
    edges_out = (uint8_t*)malloc(width * height * 1);
    size_t edges_size = gray_size;

    uint8_t* red_roads_out;
    red_roads_out = (uint8_t*)malloc(width * height * CHANNEL_NUM);
    size_t red_roads_size = rgb_size;

    VideoCapture video;
    video.open("project_video.mp4");
    Mat frame; //mat object for storing data
    uint8_t* rgb_in;
    int upper_limit = 250, lower_limit = 180;

    for (;;) {
        video >> frame;
        cvtColor(frame, frame, COLOR_BGR2RGB);
        rgb_in = frame.data;

        
        rgb2gray(rgb_in, gray_out, width, height, channels);
        threshold(gray_out, edges_out, width, height, lower_limit, upper_limit);
        thresh2lanes(rgb_in, red_roads_out, edges_out, width, height, channels);

        frame.data = red_roads_out;
        cvtColor(frame, frame, COLOR_BGR2RGB);
        imshow("frame", frame);
        if (waitKey(1) > 0) break;
    }


    free(gray_out);
    free(edges_out);
    free(red_roads_out);
    return 0;
}

void rgb2gray(uint8_t* input_img, uint8_t* gray_img, int width, int height, int channels)
{
    int gray_channels = 1;
    size_t img_size = width * height * channels;
    for (unsigned char* ptr_img = input_img, *ptr_gray = gray_img; ptr_img != input_img + img_size; ptr_img += channels, ptr_gray += gray_channels)
    {
        uint8_t red = *(ptr_img + 0);
        uint8_t green = *(ptr_img + 1);
        uint8_t blue = *(ptr_img + 2);
        uint8_t gray = 0.21f * red + 0.71f * green + 0.07 * blue;

        *ptr_gray = (uint8_t)gray;
    }
}

void threshold(uint8_t* gray_img, uint8_t* threshold_img,
    int width, int height, uint8_t lower_limit, uint8_t upper_limit)
{
    size_t img_size = width * height * 1;
    for (unsigned char* ptr_gray = gray_img, *ptr_thresh = threshold_img; ptr_gray != gray_img + img_size; ptr_gray++, ptr_thresh++)
    {
        uint8_t pixel = *ptr_gray;
        if (pixel > lower_limit && pixel < upper_limit)
            *ptr_thresh = 240;
        else
            *ptr_thresh = 0;
    }
}


void thresh2lanes(uint8_t* input_img, uint8_t* out_img, uint8_t* thresh, int width, int height, int channels)
{
    size_t img_size = width * height * channels;
    int index = 0;
    int number = 0;
    for (unsigned char* ptr_img = input_img, *ptr_out = out_img, *ptr_thresh = thresh; ptr_img != input_img + img_size; ptr_img += channels, ptr_out += channels, ptr_thresh++)
    {
        if (((index) > (height * 0.6)) && *ptr_thresh != 0)
        {
            *(ptr_out + 0) = 240;
            *(ptr_out + 1) = 0;
            *(ptr_out + 2) = 0;
        }
        else
        {
            *(ptr_out + 0) = *(ptr_img + 0);
            *(ptr_out + 1) = *(ptr_img + 1);
            *(ptr_out + 2) = *(ptr_img + 2);
        }
        number++;
        if (number % width == 0)
        {
            index++;
        }
    }
}
