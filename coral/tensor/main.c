#include "tensor.h"
#include "variable.h"
#include "debug.h"
#include <stdio.h>

int main() {
    variable_t* x = new_variable(3, 2, 2, 4);
    variable_t* y = new_variable(1, 4);
    set_to_scalar(x, 10);
    set_to_scalar(y, 2);
    variable_t* z = add(x, y);
    display_variable(x);
    display_variable(y);
    display_variable(z);
    DEBUG_ASSERT(0, "Oh no, %d\n", 3);
}