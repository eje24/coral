#include "shape.h"
#include "stdbool.h"

shape_t* _new_shape(int num_dims, size_t* dims){
    shape_t* new_shape = (shape_t*) malloc(sizeof(shape_t));
    new_shape->dims = (size_t*) malloc(num_dims * sizeof(size_t));
    new_shape->strides = (size_t*) malloc(num_dims * sizeof(size_t));
    for(int index = 0; index < num_dims; index++){
        new_shape->dims[index] = dims[index];
    }
    size_t stride = 1;
    for(int stride_index = num_dims-1; stride_index >=0; stride_index--){
        new_shape->strides[stride_index] = stride;
        stride *= new_shape->dims[stride_index];
    }
    new_shape->size = new_shape->dims[0] * new_shape->strides[0];
    return new_shape;
}

shape_t* _copy_shape(const shape_t* shape){
    return _new_shape(shape->num_dims, shape->dims);
}

shape_t* _get_broadcast_shape(const shape_t* left_shape, const shape_t* right_shape){
    if(left_shape->num_dims < right_shape->num_dims){
        return _get_broadcast_shape(right_shape, left_shape);
    }
    int num_dims = left_shape->num_dims;
    size_t dims[num_dims];
    int offset = left_shape->num_dims - right_shape->num_dims;
    for(int index = 0; index < offset; index++){
        dims[index] = left_shape->dims[index];
    }
    for(int index = offset; index < num_dims; index++){
        dims[index] = MAX(left_shape->dims[index], right_shape->dims[index-offset]);
    }
    return _new_shape(num_dims, dims);
}

bool _shape_broadcast_compatible(shape_t* left_shape, shape_t* right_shape){
    if(left_shape->num_dims < right_shape->num_dims){
        return _tensor_broadcast_componentwise_compatible(right_shape, left_shape);
    }
    // left_tensor->num_dims >= right_tensor->num_dims
    int dim_offset = left_shape->num_dims - right_shape->num_dims;
    for(int dim_index = 0; dim_index < right_shape->num_dims; dim_index++){
        size_t left_dim = left_shape->dims[dim_offset + dim_index];
        size_t right_dim = right_shape->dims[dim_index];
        if(left_dim > 1 && right_dim > 1 && left_dim != right_dim){
            return 0;
        }
    }
    return 1;
}
