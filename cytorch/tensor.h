#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

typedef float tensor_entry_t; 
typedef uint64_t tensor_size_t;

typedef struct {
    tensor_entry_t* data;
    tensor_size_t num_rows;
    tensor_size_t num_columns;
} tensor_t;

tensor_t _new_tensor(tensor_size_t num_rows, tensor_size_t num_columns);
tensor_t _new_tensor_like(tensor_t old_tensor);
tensor_t _copy_tensor(tensor_t old_tensor);

tensor_t _tensor_add(tensor_t left_tensor, tensor_t right_tensor);
inline uint8_t set_tensor_entry(tensor_t* tensor, tensor_size_t row, tensor_size_t column, tensor_entry_t value);

#endif // TENSOR_H