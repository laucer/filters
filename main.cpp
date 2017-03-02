#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include "tgaimage.h"

#define cro(x,y) (((x)+(y)-1) / y)
#define HIGH 48
#define LOW 16

using namespace std;

CUfunction kconvolution, kgray, kmagnitude, ksobel, ksuppression, kthreshold, kcc, ktranspose, kmask;

void init_cuda();
void store_img(unsigned char* red, unsigned char* green, unsigned char* blue, int w, int h);
unsigned char* convolute(unsigned char* input, int w, int h, char* convo_matrix, int convo_w, int convo_h, bool normalize);
unsigned char* gauss(unsigned char* input, int w, int h);
unsigned char* to_gray(unsigned char* red, unsigned char* green, unsigned char* blue, int w, int h);
void alternative_sobel(unsigned char* lum, unsigned char* mag, unsigned char* dir, int w, int h);
unsigned char* suppression(unsigned char* mag, unsigned char* dir, int w, int h);
unsigned char* threshold(unsigned char* mag, int w, int h, unsigned char low, unsigned char high); // 255 - strong, 127 - weak, 0 - ignore
unsigned char* hysteresis(unsigned char* magnitude, unsigned char* label, int w, int h);

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
  unsigned char* c_labels = new unsigned char[image_heigth * image_width];
  unsigned char* c_suppressed_magnitude = new unsigned char[image_heigth * image_width];
  unsigned char* c_hysteresis_magnitude = new unsigned char[image_heigth * image_width];

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
  alternative_sobel(c_luminosity, c_magnitude, c_direction, image_width, image_heigth);
  c_suppressed_magnitude = suppression(c_magnitude, c_direction, image_width, image_heigth);
  c_labels = threshold(c_suppressed_magnitude, image_width, image_heigth, LOW, HIGH);
  c_hysteresis_magnitude = hysteresis(c_suppressed_magnitude, c_labels, image_width, image_heigth);

  store_img(c_hysteresis_magnitude, c_hysteresis_magnitude, c_hysteresis_magnitude, image_width, image_heigth);

  cout << "done";
}

unsigned char* convolute(unsigned char* input, int w, int h, char* convo_matrix, int convo_w, int convo_h){
  CUdeviceptr src, res, convo;
  cuMemAlloc(&src, w * h);
  cuMemcpyHtoD(src, (const void*)input, w * h);

  cuMemAlloc(&res, w * h);

  cuMemAlloc(&convo, convo_w * convo_h);
  cuMemcpyHtoD(convo, (const void*)convo_matrix, convo_w * convo_h);

  void* args[] = {&src, &w, &h, &convo, &convo_w, &convo_h, &res};
  cuLaunchKernel(kconvolution, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();

  unsigned char* result = new unsigned char[w*h];
  cuMemcpyDtoH(result, res, w*h);

  cuMemFree(res);
  cuMemFree(src);
  cuMemFree(convo);

  return result;
}

void store_img(unsigned char* red, unsigned char* green, unsigned char* blue, int w, int h){
  TGAImage img = TGAImage(w, h, 3);
  for(int y = 0; y < h; y++){
    for(int x = 0; x < w; x++){
      TGAColor pixel = TGAColor(red[x + y * w], green[x + y * w], blue[x + y * w]);
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
  return convolute(input, w, h, convo_matrix, 5, 5);
}

void init_cuda(){
  CUdevice dev;
  CUcontext con;
  CUmodule mod;
  cuInit(0);
  cuDeviceGet(&dev, 0);
	cuCtxCreate(&con, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev);
  cuModuleLoad(&mod, "kernels.ptx");

  cuModuleGetFunction(&ksuppression, mod, "suppression");
  cuModuleGetFunction(&kconvolution, mod, "convolution");
  cuModuleGetFunction(&kmagnitude, mod, "get_magnitude");
  cuModuleGetFunction(&kthreshold, mod, "threshold");
  cuModuleGetFunction(&ktranspose, mod, "transpose");
  cuModuleGetFunction(&ksobel, mod, "sobel");
  cuModuleGetFunction(&kgray, mod, "to_gray");
  cuModuleGetFunction(&kmask, mod, "mask");
  cuModuleGetFunction(&kcc, mod, "cc");
}

unsigned char* to_gray(unsigned char* red, unsigned char* green, unsigned char* blue, int w, int h){
  CUdeviceptr dev_red, dev_green, dev_blue, dev_light;
  cuMemAlloc(&dev_red, w * h);
  cuMemAlloc(&dev_green, w * h);
  cuMemAlloc(&dev_blue, w * h);
  cuMemAlloc(&dev_light, w * h);
  cuMemcpyHtoD(dev_red, (const void*)red, w * h);
  cuMemcpyHtoD(dev_green, (const void*)green, w * h);
  cuMemcpyHtoD(dev_blue, (const void*)blue, w * h);

  void* args[] = {&dev_red, &dev_green, &dev_blue, &dev_light, &w, &h};
  cuLaunchKernel(kgray, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();

  unsigned char* result = new unsigned char[w*h];
  cuMemcpyDtoH(result, dev_light, w*h);
  return result;
}

void alternative_sobel(unsigned char* lum, unsigned char* mag, unsigned char* dir, int w, int h){
  CUdeviceptr dev_light, dev_mag, dev_dir;
  cuMemAlloc(&dev_light, w * h);
  cuMemAlloc(&dev_mag, w * h);
  cuMemAlloc(&dev_dir, w * h);
  cuMemcpyHtoD(dev_light, (const void*)lum, w * h);
  void* args[] = {&dev_light, &dev_mag, &dev_dir, &w, &h};
  cuLaunchKernel(ksobel, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();

  cuMemcpyDtoH(mag, dev_mag, w*h);
  cuMemcpyDtoH(dir, dev_dir, w*h);
}

unsigned char* suppression(unsigned char* mag, unsigned char* dir, int w, int h){
  CUdeviceptr dev_res, dev_mag, dev_dir;
  cuMemAlloc(&dev_res, w * h);
  cuMemAlloc(&dev_mag, w * h);
  cuMemAlloc(&dev_dir, w * h);
  cuMemcpyHtoD(dev_mag, (const void*)mag, w * h);
  cuMemcpyHtoD(dev_dir, (const void*)dir, w * h);
  void* args[] = {&dev_mag, &dev_dir, &dev_res, &w, &h};
  cuLaunchKernel(ksuppression, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
	cuCtxSynchronize();

  unsigned char* result = new unsigned char[w*h];
  cuMemcpyDtoH(result, dev_res, w*h);
  return result;
}

unsigned char* threshold(unsigned char* mag, int w, int h, unsigned char low, unsigned char high){ // 255 - strong, 127 - weak, 0 - ignore
  CUdeviceptr dev_mag, dev_res;
  cuMemAlloc(&dev_mag, w * h);
  cuMemAlloc(&dev_res, w * h);
  cuMemcpyHtoD(dev_mag, (const void*)mag, w * h);
  void* args[] = {&dev_mag, &dev_res, &low, &high, &w, &h};
  cuLaunchKernel(kthreshold, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);

  unsigned char* result = new unsigned char[w*h];
  cuMemcpyDtoH(result, dev_res, w*h);
  return result;
}

unsigned char* hysteresis(unsigned char* magnitude, unsigned char* label, int w, int h){
  CUdeviceptr dev_label, dev_label_trans, dev_mag;
  cuMemAlloc(&dev_label, w * h);
  cuMemAlloc(&dev_mag, w * h);
  cuMemAlloc(&dev_label_trans, w * h);
  cuMemcpyHtoD(dev_label, (const void*)label, w * h);
  cuMemcpyHtoD(dev_mag, (const void*)magnitude, w * h);

  void* args[] = {&dev_label, &w, &h};
  cuLaunchKernel(kcc, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args, 0);
  cuCtxSynchronize();

  void* args_transpose[] = {&dev_label, &dev_label_trans, &w, &h};
  cuLaunchKernel(ktranspose, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args_transpose, 0);
  cuCtxSynchronize();

  void* args2[] = {&dev_label_trans, &h, &w};
  cuLaunchKernel(kcc, cro(h,1024), 1, 1, 1024, 1, 1, 0, 0, args2, 0);
  cuCtxSynchronize();

  void* args_transpose2[] = {&dev_label_trans, &dev_label, &h, &w};
  cuLaunchKernel(ktranspose, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args_transpose2, 0);
  cuCtxSynchronize();

  void* args_mask[] = {&dev_mag, &dev_label, &w, &h};
  cuLaunchKernel(kmask, cro(w,32), cro(h,32), 1, 32, 32, 1, 0, 0, args_mask, 0);
  cuCtxSynchronize();

  unsigned char* result = new unsigned char[w*h];
  cuMemcpyDtoH(result, dev_mag, w*h);
  return result;
}
