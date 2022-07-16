#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdint.h>
#include <math.h>
#include <qstring.h>

#define FACTOR 0.5
#define WIDTH 1280
#define HEIGHT 720
#define CHANNELS 3

/*=============== Qt Versions =================*/

QString rgb_2_gray(uint8_t* d_gray_out, uint8_t* d_rgb_in);

QString doub_threshold(uint8_t* out_img, uint8_t* in_img,
    uint8_t lower_limit, uint8_t upper_limit);

QString thresh_2_lanes(uint8_t* red_roads_img, uint8_t* edges_img);