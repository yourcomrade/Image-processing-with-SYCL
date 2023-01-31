//This file is to convert RGB to a binary image
//using parallel_for and USM style with CPU
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include "CImg.h"
using namespace cimg_library;
class my_kernel;
void say_device(const sycl::queue&q){
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>()
       << "\n";
}
int main(){
    sycl::cpu_selector device;
    sycl::queue Q(device);
    say_device(Q);
    CImg<unsigned char> image("Lenna.bmp");
    int w = image.width();
    int h = image.height();
    //Get pixel array from image
    unsigned char*ptr = image.data();
    //Get Red pixel array from image
    unsigned char *host_R = &ptr[0]; 
    //Get Green pixel array from image
    unsigned char *host_G = &ptr[0+h*w];
    //Get Blue pixel array from image
    unsigned char *host_B = &ptr[0+2*h*w];
    //Allocate device memory
    auto dev_R = sycl::malloc_device<unsigned char>(h*w,Q);
    auto dev_G = sycl::malloc_device<unsigned char>(h*w,Q);
    auto dev_B = sycl::malloc_device<unsigned char>(h*w,Q);
    
    auto start = std::chrono::steady_clock::now();
    //Copy data from host to device
    Q.memcpy(dev_R,host_R,h*w*sizeof(unsigned char));
      Q.memcpy(dev_G,host_G,h*w*sizeof(unsigned char));
      Q.memcpy(dev_B,host_B,h*w*sizeof(unsigned char));
    Q.submit([&](sycl::handler &cgh){
     
     cgh.parallel_for(sz,[=](sycl::id<1>i){
        dev_R[i] =((dev_R[i]+dev_G[i]+dev_B[i])/3>123)?255:0; ;
        dev_G[i] = dev_R[i];
        dev_B[i] = dev_R[i];
     });
     
   }).wait();
    Q.memcpy(host_R,dev_R,h*w*sizeof(unsigned char));
      Q.memcpy(host_G,dev_G,h*w*sizeof(unsigned char));
      Q.memcpy(host_B,dev_B,h*w*sizeof(unsigned char));
      Q.wait();
   auto end = std::chrono::steady_clock::now();
   std::cout<<"Kernel submission + execution time: "
   <<std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
   <<" microsec\n";
   //Deallocate device memory
   sycl::free(dev_R,Q);
   sycl::free(dev_G,Q);
   sycl::free(dev_B,Q);
   image.save("binary.bmp");
   return 0;
}