#include<cstdio>
#include "vector_types.h"
#define pi 3.14159265359
#define pos(x,y) ((x) + (y)*w)

extern "C" {

__global__ void convolution(unsigned char* src, int w, int h, char* convo, int convo_w, int convo_h, unsigned char* dst){
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
	}
	int val = computed_weigth > 0 ? computed_pixel / computed_weigth : computed_pixel;
	dst[pos(x,y)] = val;

}


// l - light
__global__ void sobel(unsigned char* l, unsigned char* magnitude, unsigned char* direction, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w - 2 || y >= h - 2) return;
	if (x < 1 || y < 1) return;

	int dx = 0
	+ l[pos(x-1, y-1)] - l[pos(x+1, y-1)]
	+ 2*l[pos(x-1, y)] - 2*l[pos(x+1, y)]
	+ l[pos(x-1, y+1)] - l[pos(x+1, y+1)];

	int dy = 0
	+ l[pos(x-1, y-1)] + 2*l[pos(x, y-1)] + l[pos(x+1, y-1)]
	- l[pos(x-1, y+1)] - 2*l[pos(x, y+1)] - l[pos(x+1, y+1)];

	int mag = sqrt((double)dx*dx + dy*dy);
	int dir = atan((double)dy/dx);

	magnitude[x + y*w] = mag;
	int degree = dir * 180.0 / pi;
	if (degree < 0) degree += 180;
	direction[x+ y*w] = degree;
}

__global__ void suppression(unsigned char* magnitude, unsigned char* direction, unsigned char* result_magnitude, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w - 2 || y >= h - 2) return;
	if (x < 1 || y < 1) return;

	int degree = direction[x + y * w];
	int mag = magnitude[x + y * w];
	bool greatest = false;
	if(degree < 22 || degree > 180 - 22){ // poziomo
		if(mag > magnitude[pos(x-1, y)] && mag > magnitude[pos(x+1, y)]) greatest = true;
	} else if (degree < 45+22){ // gora prawo
		if(mag > magnitude[pos(x+1, y+1)] && mag > magnitude[pos(x-1, y-1)]) greatest = true;
	} else if (degree < 90 + 22) { // pionowo
		if(mag > magnitude[pos(x,y+1)] && mag > magnitude[pos(x, y-1)]) greatest = true;
	} else { // gora lewo
		if(mag > magnitude[pos(x-1,y+1)] && mag > magnitude[pos(x+1, y-1)]) greatest = true;
	}

	if(!greatest){
		mag = 0;
	}
	result_magnitude[x + y * w] = mag;

}

__global__ void threshold(unsigned char* magnitude, unsigned char* label, unsigned char low, unsigned char high, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int value = magnitude[pos(x,y)];
	if(value >= high) label[pos(x,y)] = 255;
	else if (value >= low) label[pos(x,y)] = 127;
	else label[pos(x,y)] = 0;
}

__global__ void mask(unsigned char* magnitude, unsigned char* mask, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int value = mask[pos(x,y)];
	if(value!=255) magnitude[pos(x,y)] = 0;
}

__global__ void transpose(unsigned char* in, unsigned char* out, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	out[y + x * h] = in[pos(x,y)];
}

__global__ void cc(unsigned char* label, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= w) return;
	for(int y = 1; y < h; y++){
		if(label[pos(x, y-1)] == 255 && label[pos(x,y)] == 127) label[pos(x,y)] = 255;
	}
	for(int y = h - 1; y > 0; y--){
		if(label[pos(x, y)] == 255 && label[pos(x, y-1)] == 127) label[pos(x, y-1)] = 255;
	}
}

__global__ void to_gray(unsigned char* red, unsigned char* green, unsigned char* blue, unsigned char* dst, int w, int h){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	int r = 0.2989 * red[pos(x,y)];
	int g = 0.5870 * green[pos(x,y)];
	int b = 0.1140 * blue[pos(x,y)];
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
