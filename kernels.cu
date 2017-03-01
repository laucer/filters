#include<cstdio>
#include "vector_types.h"
#define pi 3.14159265359
extern "C" {

__global__ void convolution(unsigned char* src, int w, int h, char* convo, int convo_w, int convo_h, unsigned char* dst, bool normalize){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	__shared__ int shr_convo[32][32];
	if(threadIdx.x == 0 && threadIdx.y == 0){
		int p = 0;
		for(int i = 0; i < convo_w; i++){
			for(int j = 0; j < convo_h; j++){
				shr_convo[i][j] = convo[p++];
			}
		}
	} // end convo load
    __syncthreads();
		int computed_pixel = 0;
	int computed_weigth = 0;
	int radius_x = convo_w / 2, radius_y = convo_h / 2;
	int s_x = x - radius_x, s_y = y - radius_y;

	for(int i = 0; i < convo_w; i++){
		for(int j = 0; j < convo_h; j++){
			if(s_x >= 0 && s_x < w && s_y >= 0 && s_y < h){
				computed_pixel += shr_convo[i][j] * src[s_x + s_y * w];
				computed_weigth += shr_convo[i][j];
			}
			s_y++;
		}
		s_y -= convo_h;
		s_x++;
	}/*
	int ppp = 0;
	for(int yyy = y - radius_y; yyy <= y + radius_y; yyy++){
		for(int xxx = x - radius_x; xxx <= x + radius_x; xxx++){
			if(xxx < 0 || xxx >= w || yyy < 0 || yyy >= h){
				ppp++;
				continue;
			}
			computed_weigth += convo[ppp];
			computed_pixel += convo[ppp++] * src[xxx + yyy*w];
		}
	}*/
	if(normalize){
		int val = computed_weigth > 0 ? computed_pixel / computed_weigth : computed_pixel;
		dst[x + y * w] = val;
	} else{
		dst[x + y * w] = computed_pixel;
	}
}

__global__ void sobel(unsigned char* l, unsigned char* magnitude, unsigned char* direction, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w - 3 || y >= h - 3) return;
	if (x < 2 || y < 2) return;

	int dx = 0
	+ l[(x - 1) + w * (y - 1)] - l[(x + 1) + w * (y - 1)]
	+ 2*l[(x - 1) + w * (y + 0)] - 2*l[(x + 1) + w * (y + 0)]
	+ l[(x - 1) + w * (y + 1)] - l[(x + 1) + w * (y + 1)];

	int dy = 0
	+ l[(x - 1) + w * (y - 1)] + 2*l[(x + 0) + w * (y - 1)] + l[(x + 1) + w * (y - 1)]
	- l[(x - 1) + w * (y + 1)] - 2*l[(x + 0) + w * (y + 1)] - l[(x + 1) + w * (y + 1)];

	int mag = sqrt((double)dx*dx + dy*dy);
	int dir = atan((double)dy/dx);

	magnitude[x + y*w] = mag;
	direction[x+ y*w] = dir * 180.0 / pi;
}

__global__ void to_gray(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* dst, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int r = 0.2989 * red[x + y * w];
	int g = 0.5870 * green[x + y * w];
	int b = 0.1140 * blue[x + y * w];
	dst[x + y * w] = r+g+b;

}















///// legace
__global__ void get_magnitude(unsigned char* l1, unsigned char* l2, unsigned char* dst, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int v1 = l1[x + y * w];
	int v2 = l2[x + y * w];
	int val = sqrt((float)v1 * v1 + v2 * v2);
	dst[x + y * w] = val;
}

}
