#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <utils.h>

typedef struct {
    int num_dims;
    size_t size;
    size_t* dims;
    size_t* strides;
} shape_t;

shape_t* _new_shape(int num_dims, size_t* dims);
shape_t* _copy_shape(shape_t* shape);
shape_t* _get_broadcast_shape(const shape_t* left_shape, const shape_t* right_shape);
shape_t* _extend_shape_to_dims(const shape_t* shape, int num_dims);

static inline bool _shape_is_scalar(shape_t* shape){
    return shape->size == 1;
}