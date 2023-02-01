#include "shape.h"
#include "assert.h"
#include <stdbool.h>


shape_t* shape_new(int num_dims, size_t* dims){
    shape_t* new_shape = (shape_t*) malloc(sizeof(shape_t));
    new_shape->num_dims = num_dims;
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

shape_t* shape_copy(shape_t* shape){
    return shape_new(shape->num_dims, shape->dims);
}

bool shape_equal(shape_t* left_shape, shape_t* right_shape){
    if((left_shape->size != right_shape->size) || (left_shape->num_dims != right_shape->num_dims)){
        return 0;
    }
    // otherwise, both shapse have the same size and dimensions
    for(int dim_index = 0; dim_index < left_shape->num_dims; dim_index++){
        if(left_shape->dims[dim_index] != right_shape->dims[dim_index]){
            return 0;
        }
    }
    return 1;
}

shape_t* shape_get_broadcast_shape(shape_t* left_shape, shape_t* right_shape){
    if(left_shape->num_dims < right_shape->num_dims){
        return shape_get_broadcast_shape(right_shape, left_shape);
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
    return shape_new(num_dims, dims);
}

bool shape_broadcast_compatible(shape_t* left_shape, shape_t* right_shape){
    if(left_shape->num_dims < right_shape->num_dims){
        return shape_broadcast_compatible(right_shape, left_shape);
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

// packs the shape to num_dims number of dimensions
shape_t* shape_extend_to_dims(shape_t* shape, int num_dims){
    NDEBUG_ASSERT(shape->num_dims <= num_dims, "Cannot reduce the number of dimensions of a shape.\n");
    int num_new_dims = num_dims - shape->num_dims;
    size_t dims[num_dims];
    for(int index = 0; index < num_new_dims; index++){
        dims[index] = 1;
    }
    for(int index = num_new_dims; index < num_dims; index++){
        dims[index] = shape->dims[index - num_new_dims];
    }
    return shape_new(num_dims, dims);
}

void shape_verbose_display(shape_t* shape){
    printf("Size: %zu\n", shape->size);
    printf("Dimensions: ");
    printf("%zu", shape->dims[0]);
    for(int index = 1; index < shape->num_dims; index++){
        printf(" x %zu", shape->dims[index]);
    }
    printf("\n");
    printf("Strides: ");
    printf("%zu", shape->strides[0]);
    for(int index = 1; index < shape->num_dims; index++){
        printf(" x %zu", shape->strides[index]);
    }
    printf("\n");
}

void shape_display(shape_t* shape){
    printf("Dimensions: ");
    printf("%zu", shape->dims[0]);
    for(int index = 1; index < shape->num_dims; index++){
        printf(" x %zu", shape->dims[index]);
    }
    printf("\n");
}
