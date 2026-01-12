//https://blog.csdn.net/martian665/article/details/146975958
//!!!!!             原贴的代码有很多逻辑问题，感觉四不像，可能是现在学习的还不深入，没有彻底理解代码。



//Particle system simulation

//eg:

#include <cuda_runtime.h>
#include "stdio.h"
#include "stdlib.h"
//#include "time.h"
#include "curand_kernel.h"



struct Particle{
    float3 position;
    float3 velocity;
    float3 color;
};

typedef struct Particle par;


//update the state of Particle
//Apply gravity into velocity
__global__ void UpateParticle(par* particle, int n , float stride, float loop){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < n){
        for(float i = 0.0f; i < loop; i+=stride){

            particle[idx].velocity.y += -9.81f * stride;
            particle[idx].position.x += particle[idx].velocity.x * stride; 
            particle[idx].position.y += particle[idx].velocity.y * stride;
            particle[idx].position.z += particle[idx].velocity.z * stride;


            if(particle[idx].position.y < 0.0f){
                particle[idx].position.y = 0.0f;
                particle[idx].velocity.y *= -0.5f; // energy loss
            }
        }


    }

}

__global__ void stateInitialization(par* Particle, int nums){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(idx < nums){
        curandState state;
        curand_init(clock64(), idx, 0, &state);
        Particle[idx].position.x = curand_uniform(&state);
        Particle[idx].position.y = curand_uniform(&state);
        Particle[idx].position.z = curand_uniform(&state);

        Particle[idx].velocity.x = curand_uniform(&state);
        Particle[idx].velocity.y = curand_uniform(&state);
        Particle[idx].velocity.z = curand_uniform(&state);

        Particle[idx].color.x = 0.0f;
        Particle[idx].color.y = 0.0f;
        Particle[idx].color.z = 0.0f;
    }
}

int main(){
    float  stride = 0.0001f; //0.0001s
    //原帖引入的是delta time, 但是delta time常用于游戏引擎。
    //或许delta time可能会在一些特定的物理模拟中引入，但是当前代码中引入无意义。
    float total_time = 1.f;
    int nums_par = 10000;

    Particle* par = (Particle*)malloc(sizeof(Particle) * nums_par);
    Particle* dpar;
    cudaMalloc((void**)&dpar, sizeof(Particle) * nums_par);


    int block_Num;
    int block_Size;


    block_Size = 100;
    block_Num = (nums_par + block_Size - 1) / block_Size;
    stateInitialization<<<block_Num,block_Size>>>(dpar,nums_par);

    cudaDeviceSynchronize();

    UpateParticle<<<block_Num,block_Size>>>(dpar, nums_par, stride, total_time);

    cudaDeviceSynchronize();

    cudaMemcpy(par,dpar,sizeof(Particle) * nums_par,cudaMemcpyDeviceToHost);

    for(int i =0; i < nums_par; i++){
        printf("%f\n",par[i].position.y);

    }
    

    return 0;
}

