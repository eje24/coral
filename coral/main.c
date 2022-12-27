#include "tensor.h"
#include "variable.h"
#include "debug.h"
#include <stdio.h>

int main() {
    variable_t* my_variable = new_variable(2, 2, 4);
    printf("dims: %d\n", my_variable->tensor->num_dims);
    display_variable(my_variable);
    DEBUG_ASSERT(0, "Oh no, %d\n", 3);
}