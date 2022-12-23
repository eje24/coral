#include "tensor.h"
#include "variable.h"
#include "debug.h"

int main() {
    variable_t my_variable = new_variable(4, 5);
    DEBUG_ASSERT(0, "Oh no, %d\n", 3);
}