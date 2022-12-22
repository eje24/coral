#include "tensor.h"
#include "variable.h"
#include "basic_functions.h"
#include "utils.h"

#include <stdbool.h>

// performs component-wise addition
variable_t add(variable_t left_variable, variable_t right_variable){
    if(!dimensions_match(left_variable.tensor, right_variable.tensor)){
        fprintf(OUT, "Error: cannot add tensor of sizes (%lu,%lu) and (%lu,%lu).\n",
            left_variable.tensor.num_rows, left_variable.tensor.num_columns,
            right_variable.tensor.num_rows, right_variable.tensor.num_columns);
    }
    tensor_t new_tensor = _tensor_add(left_variable.tensor, right_variable.tensor);
    return _new_variable(new_tensor);
}