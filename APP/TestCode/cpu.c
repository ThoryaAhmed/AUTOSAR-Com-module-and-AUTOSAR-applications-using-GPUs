// Load image using the stb_image library, convert the image to gray and sepia, write it back to disk
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <time.h>
#define REPS 5

void rgb2gray(uint8_t *input_img, uint8_t *gray_img, int width, int height, int channels)
{
    int gray_channels = 1;
    size_t img_size = width * height * channels;
    for (unsigned char *ptr_img = input_img, *ptr_gray = gray_img; ptr_img != input_img + img_size; ptr_img += channels, ptr_gray += gray_channels)
    {
        uint8_t red = *(ptr_img + 0);
        uint8_t green = *(ptr_img + 1);
        uint8_t blue = *(ptr_img + 2);
        uint8_t gray = 0.21f * red + 0.71f * green + 0.07 * blue;

        *ptr_gray = (uint8_t)gray;
    }
}

void threshold(uint8_t *gray_img, uint8_t *threshold_img,
               int width, int height, uint8_t lower_limit, uint8_t upper_limit)
{
    size_t img_size = width * height * 1;
    for (unsigned char *ptr_gray = gray_img, *ptr_thresh = threshold_img; ptr_gray != gray_img + img_size; ptr_gray++, ptr_thresh++)
    {
        uint8_t pixel = *ptr_gray;
        if (pixel > lower_limit && pixel < upper_limit)
            *ptr_thresh = 240;
        else
            *ptr_thresh = 0;
    }
}

void thresh2lanes(uint8_t *org_img, uint8_t *edges_img, uint8_t *roads_img,
                  int width, int height, int channels)
{

    size_t img_size = width * height * 1;
    for (uint8_t *ptr_edges = edges_img, *ptr_roads = roads_img, *ptr_org = org_img; ptr_edges != ptr_edges + img_size; ptr_edges++, ptr_roads += channels, ptr_org += channels)
    {
        if (*ptr_edges != 0)
        {
            *(ptr_roads + 0) = 250; // red
            *(ptr_roads + 1) = 0;   // no green
            *(ptr_roads + 2) = 0;   // no blue
        }
        else
        {
            *(ptr_roads + 0) = *(ptr_org + 0);
            *(ptr_roads + 1) = *(ptr_org + 1);
            *(ptr_roads + 2) = *(ptr_org + 2);
        }

        // if(*ptr_roads == 250)
        // printf("*\n");
    }
}

void rgb2rgb(uint8_t *input_img, uint8_t *out_img, uint8_t *thresh, int width, int height, int channels)
{
    size_t img_size = width * height * channels;
    int index = 0;
    int number = 0;
    for (unsigned char *ptr_img = input_img, *ptr_out = out_img, *ptr_thresh = thresh; ptr_img != input_img + img_size; ptr_img += channels, ptr_out += channels, ptr_thresh++)
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

int main(void)
{
    clock_t functions_time[3];
    clock_t Ti, Tf;
    double time_taken;

    int width, height, channels;
    unsigned char *img = stbi_load("image.jpg", &width, &height, &channels, 0);
    if (img == NULL)
    {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf("Loaded image with a width of %dpx, a height of %dpx and %d channels\n", width, height, channels);
    // Convert the input image to gray
    size_t img_size = width * height * channels;

    int gray_channels = 1;
    size_t gray_img_size = width * height * gray_channels;
    unsigned char *gray_img = (unsigned char *)malloc(gray_img_size);
    if (gray_img == NULL)
    {
        printf("Unable to allocate memory for the image.\n");
        exit(1);
    }

    Ti = clock();
    for (int i = 0; i < REPS; i++)
        rgb2gray(img, gray_img, width, height, channels);
    Tf = clock();
    time_taken = (((double)(Tf - Ti)) / CLOCKS_PER_SEC) / REPS;
    printf("gray took %f seconds to execute \n", time_taken);

    stbi_write_jpg("gray.jpg", width, height, gray_channels, gray_img, 100);

    size_t threshold_img_size = width * height * 1;
    unsigned char *threshold_img = (unsigned char *)malloc(threshold_img_size);
    if (threshold_img == NULL)
    {
        printf("Unable to allocate memory for the image.\n");
        exit(1);
    }
    int upper_limit = 250, lower_limit = 180;

    Ti = clock();
    for (int i = 0; i < REPS; i++)
        threshold(gray_img, threshold_img, width, height, lower_limit, upper_limit);
    Tf = clock();
    time_taken = (((double)(Tf - Ti)) / CLOCKS_PER_SEC) / REPS;
    printf("threshold took %f seconds to execute \n", time_taken);

    stbi_write_jpg("thresh.jpg", width, height, 1, threshold_img, 100);

    size_t roads_img_size = width * height * channels;
    unsigned char *red_roads_img = (unsigned char *)malloc(roads_img_size);

    // thresh2lanes(img, threshold_img, red_roads_img, width, height, channels);
    Ti = clock();
    for (int i = 0; i < REPS; i++)
        rgb2rgb(img, red_roads_img, threshold_img, width, height, channels);
    Tf = clock();
    time_taken = (((double)(Tf - Ti)) / CLOCKS_PER_SEC) / REPS;
    printf("rgb2gray took %f seconds to execute \n", time_taken);
    printf("Done.\n");

    stbi_write_jpg("lanes.jpg", width, height, channels, red_roads_img, 100);

    stbi_image_free(img);
    free(gray_img);
    free(threshold_img);
    free(red_roads_img);
}
