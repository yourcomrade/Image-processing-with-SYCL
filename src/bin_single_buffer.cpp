//This file is to convert RGB to a binary image
//using single task and Buffer style with CPU
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
constexpr int sz = 512*512;
int main(){
    
    CImg<unsigned char> image("Lenna.bmp");
    int w = image.width();
    int h = image.height();
    std::cout<<"w: "<<w<<" h: "<<h<<"\n";
    //Get pixel matrix from image
    unsigned char*ptr = image.data();
    //Get Red pixel matrix from image
    unsigned char *host_R = &ptr[0]; 
    //Get Green pixel matrix from image
    unsigned char *host_G = &ptr[0+h*w];
    //Get Blue pixel matrix from image
    unsigned char *host_B = &ptr[0+2*h*w];
    auto start = std::chrono::steady_clock::now();
   
    {
    sycl::cpu_selector device;
    sycl::queue Q(device);
    say_device(Q);
    //Create a buffer
    auto bufR = sycl::buffer<unsigned char,1>{host_R,sycl::range{ sz}};
    auto bufG = sycl::buffer<unsigned char,1>{host_G,sycl::range{sz}};
    auto bufB = sycl::buffer<unsigned char,1>{host_B,sycl::range{sz}};
    Q.submit([&](sycl::handler &cgh){
            //Access to buffer from device side in read_write mode
            sycl::accessor R{bufR, cgh, sycl::read_write};
            sycl::accessor G{bufG, cgh, sycl::read_write};
            sycl::accessor B{bufB, cgh, sycl::read_write};
            cgh.single_task<my_kernel>([=]()
      {
                for(int i=0;i<sz;i++){
                    R[i] = ((R[i]+G[i]+B[i])/3>123)?255:0;
                    G[i] = R[i];
                    B[i] = R[i];
                }
      });
     
   }).wait();
    }
    //After go out of scope, data from buffer will be copied back to data in host
   auto end = std::chrono::steady_clock::now();
   std::cout<<"Kernel submission + execution time: "
   <<std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
   <<" microsec\n";
   image.save("binary.bmp");
    return 0;
}