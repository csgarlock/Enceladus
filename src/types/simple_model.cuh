#ifndef SIMPLE_MODEL_H_INCLUDED
#define SIMPLE_MODEL_H_INCLUDED

#include "tensor.cuh"

class SimpleModel {

    public:

    Tensor<float> masses;
    Tensor<float3> positions;
    Tensor<float3> velocities;

    float time_step;
    

};

#endif