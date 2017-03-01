#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#include "tgaimage.h"

using namespace std;

#define cro(x,y) (((x)+(y)-1) / y)

CUfunction kconvolution, kgray, kmagnitude, ksobel, ksuppression;

void init_cuda();
void store_img(unsigned char* red, unsigned char* green, unsigned char* blue, int w, int h);
unsigned char* convolute(unsigned char* input, int w, int h, char* convo_matrix, int convo_w, int convo_h, bool normalize);
unsigned char* gauss(unsigned char* input, int w, int h);
unsigned char* to_gray(unsigned char* red, unsigned char* green, unsigned char* blue, int w, int h);
void alternative_sobel(unsigned char* lum, unsigned char* mag, unsigned char* dir, int w, int h);
unsigned char* suppression(unsigned char* mag, unsigned char* dir, int w, int h);



int main(){
  init_cuda();
  TGAImage image = TGAImage();
  image.read_tga_file("marbles.TGA");

  int image_width = image.get_width();
  int image_heigth = image.get_height();
  unsigned char* c_red = new unsigned char[image_heigth * image_width];
  unsigned char* c_green = new unsigned char[image_heigth * image_width];
  unsigned char* c_blue = new unsigned char[image_heigth * image_width];

  unsigned char* c_luminosity = new unsigned char[image_heigth * image_width];
  unsigned char* c_magnitude = new unsigned char[image_heigth * image_width];
  unsigned char* c_direction = new unsigned char[image_heigth * image_width];
  unsigned char* c_suppressed = new unsigned char[image_heigth * image_width];



  for(int y = 0; y < image_heigth; y++){
    for(int x = 0; x < image_width; x++){
      TGAColor pixel = image.get(x, y);
      c_blue[x + y*image_width] = pixel.bgra[0];
      c_green[x + y*image_width] = pixel.bgra[1];
      c_red[x + y*image_width] = pixel.bgra[2];
    }
  }

  c_luminosity = to_gray(c_red, c_green, c_blue, image_width, image_heigth);

  c_luminosity = gauss(c_luminosity, image_width, image_heigth);
  //c_blue = sobel(c_blue, image_width, image_heigth);
  alternative_sobel(c_luminosity, c_magnitude, c_direction, image_width, image_heigth);
  c_suppressed = suppression(c_magnitude, c_direction, image_width, image_heigth);

  store_img(c_suppressed, c_suppressed, c_suppressed, image_width, image_heigth);

}


unsigned char* convolute(unsigned char* input, int w, int h, char* convo_matrix, int convo_w, int convo_h, bool normalize = false){
  CUdeviceptr src, dst, convo;
  cuMemAlloc(&src, w * h);
  cuMemcpyHtoD(src, (const void*)input, w * h);

  cuMemAlloc(&dst, w * h);

  cuMemAlloc(&convo, convo_w * convo_h);
  cuMemcpyHtoD(convo, (const void*)convo_matrix, convo_w * convo_h);

  void* args[] = {&src, &w, &h, &convo, &convo_w, &convo_h, &dst, &normalize};
  cuLaunchKernel(kconvolution, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();

  unsigned char* result = new unsigned char[w*h];

  cuMemcpyDtoH(result, dst, w*h);

  cuMemFree(dst);
  cuMemFree(src);
  cuMemFree(convo);

  return result;
}

void store_img(unsigned char* red, unsigned char* green, unsigned char* blue, int w, int h){
  TGAImage img = TGAImage(w, h, 3);
  for(int y = 0; y < h; y++){
    for(int x = 0; x < w; x++){
      TGAColor pixel = TGAColor(red[x + y*w], green[x + y*w], blue[x + y*w]);
      img.set(x,y,pixel);
    }
  }
  img.write_tga_file("out.TGA");
}

unsigned char* gauss(unsigned char* input, int w, int h){
  char convo_matrix[] = {  2, 4, 5, 4, 2,
                                    4, 9,12, 9, 4,
                                    5,12,15,12, 5,
                                    4, 9,12, 9, 4,
                                    2, 4, 5, 4, 2};
  return convolute(input, w, h, convo_matrix, 5, 5, true);
}

void init_cuda(){
  CUdevice d;
  CUcontext c;
  CUmodule m;
  cuInit(0);
  cuDeviceGet(&d, 0);
	cuCtxCreate(&c, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, d);
  cuModuleLoad(&m, "kernels.ptx");

  cuModuleGetFunction(&kconvolution, m, "convolution");
  cuModuleGetFunction(&kgray, m, "to_gray");
  cuModuleGetFunction(&kmagnitude, m, "get_magnitude");
  cuModuleGetFunction(&ksobel, m, "sobel");
  cuModuleGetFunction(&ksuppression, m, "suppression");
}

unsigned char* to_gray(unsigned char* red, unsigned char* green, unsigned char* blue, int w, int h){
  CUdeviceptr r, g, b, l;
  cuMemAlloc(&r, w * h);
  cuMemAlloc(&g, w * h);
  cuMemAlloc(&b, w * h);
  cuMemAlloc(&l, w * h);
  cuMemcpyHtoD(r, (const void*)red, w * h);
  cuMemcpyHtoD(g, (const void*)green, w * h);
  cuMemcpyHtoD(b, (const void*)blue, w * h);

  void* args[] = {&r, &g, &b, &l, &w, &h};
  cuLaunchKernel(kgray, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();

  unsigned char* result = new unsigned char[w*h];
  cuMemcpyDtoH(result, l, w*h);
  return result;
}

void alternative_sobel(unsigned char* lum, unsigned char* mag, unsigned char* dir, int w, int h){
  CUdeviceptr l, m, d;
  cuMemAlloc(&l, w * h);
  cuMemAlloc(&m, w * h);
  cuMemAlloc(&d, w * h);
  cuMemcpyHtoD(l, (const void*)lum, w * h);
  void* args[] = {&l, &m, &d, &w, &h};
  cuLaunchKernel(ksobel, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();

  cuMemcpyDtoH(mag, m, w*h);
  cuMemcpyDtoH(dir, d, w*h);
}

unsigned char* suppression(unsigned char* mag, unsigned char* dir, int w, int h){
  CUdeviceptr r, m, d;
  cuMemAlloc(&r, w * h);
  cuMemAlloc(&m, w * h);
  cuMemAlloc(&d, w * h);
  cuMemcpyHtoD(m, (const void*)mag, w * h);
  cuMemcpyHtoD(d, (const void*)dir, w * h);
  void* args[] = {&m, &d, &r, &w, &h};
  cuLaunchKernel(ksuppression, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();


  unsigned char* result = new unsigned char[w*h];
  cuMemcpyDtoH(result, r, w*h);
  return result;
}
















///////////////// legacy

unsigned char* sobelX(unsigned char* input, int w, int h);
unsigned char* sobelY(unsigned char* input, int w, int h);
unsigned char* sobel_magnitude(unsigned char* x, unsigned char* y, int w, int h);
unsigned char* sobel(unsigned char* lum, int w, int h);



unsigned char* sobel(unsigned char* lum, int w, int h){
  unsigned char* xx = sobelX(lum, w, h);
  unsigned char* yy = sobelY(lum, w, h);
  return sobel_magnitude(xx, yy, w, h);
}
unsigned char* sobel_magnitude(unsigned char* x, unsigned char* y, int w, int h){
  CUdeviceptr devx, devy, mag;
  cuMemAlloc(&devx, w * h);
  cuMemAlloc(&devy, w * h);
  cuMemAlloc(&mag, w * h);
  cuMemcpyHtoD(devx, (const void*)x, w * h);
  cuMemcpyHtoD(devy, (const void*)y, w * h);

  void* args[] = {&devx, &devy, &mag, &w, &h};
  cuLaunchKernel(kmagnitude, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();

  unsigned char* result = new unsigned char[w*h];
  cuMemcpyDtoH(result, mag, w*h);
  return result;
}

unsigned char* sobelX(unsigned char* input, int w, int h){
  char convo_matrix[] = {  1, 0, -1,
    2, 0, -2,
    1, 0, -1 };
  return convolute(input, w, h, convo_matrix, 3, 3);
}

unsigned char* sobelY(unsigned char* input, int w, int h){
  char convo_matrix[] = { 1, 2, 1,
  0, 0, 0,
  -1, -2, -1};
  return convolute(input, w, h, convo_matrix, 3, 3);
}
