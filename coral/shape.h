#ifndef SHAPE_H
#define SHAPE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "utils.h"

typedef struct {
    int num_dims;
    size_t size;
    size_t* dims;
    size_t* strides;
} shape_t;

shape_t* shape_new(int num_dims, size_t* dims);
shape_t* shape_copy(shape_t* shape);
bool shape_equal(shape_t* left_shape, shape_t* right_shape);
shape_t* shape_get_broadcast_shape(shape_t* left_shape, shape_t* right_shape);
bool shape_broadcast_compatible(shape_t* left_shape, shape_t* right_shape);
shape_t* shape_extend_to_dims(shape_t* shape, int num_dims);
void shape_verbose_display(shape_t* shape);
void shape_display(shape_t* shape);

static inline bool shape_is_scalar(shape_t* shape){
    return shape->size == 1;
}

#endif // SHAPE_H