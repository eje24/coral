#include "grad.h"
#include "variable.h"
#include "assert.h"

static inline void increment_ref_count(variable_t* variable){
    variable->grad_meta->ref_count++;
}

static inline void decrement_ref_count(variable_t* variable){
    variable->grad_meta->ref_count--;
}

static inline int get_ref_count(variable_t* variable){
    return variable->grad_meta->ref_count;
}

// propogate gradient update from output into input
// here, output = fn(input)
static void update_unary_grad(input_t* input, variable_t* output){
    variable_unary_grad_op_t gradient_fn = (variable_unary_grad_op_t) (input->grad_op);
    tensor_t* gradient_update = (*gradient_fn)(input->variable, output);
    tensor_t* reduced_gradient_update = tensor_reduce_to_shape(gradient_update, input->variable->gradient->shape);
    tensor_in_place_add(input->variable->gradient, reduced_gradient_update);
    decrement_ref_count(input->variable);
}

// propogate gradient update from output into input
// here, output = fn(input, other_input)
static void update_binary_grad(input_t* input, input_t* other_input, variable_t* output){
    variable_binary_grad_op_t gradient_fn = (variable_binary_grad_op_t) (input->grad_op);
    tensor_t* gradient_update = (*gradient_fn)(input->variable, other_input->variable, output);
    tensor_t* reduced_gradient_update = tensor_reduce_to_shape(gradient_update, input->variable->gradient->shape);
    printf("REDUCED GRADIENT:\n\n");
    tensor_display(reduced_gradient_update);
    tensor_in_place_add(input->variable->gradient, reduced_gradient_update);
    decrement_ref_count(input->variable);
}

// accumulate gradient updates into argument gradients
static inline void update_binary_grads(input_t* left_input, input_t* right_input, variable_t* output){
    update_binary_grad(left_input, right_input, output);
    update_binary_grad(right_input, left_input, output);
}

// accumulates gradient update into argument(s') gradient
// for arguments with ref_count zero, call _backwards an arguments
// so as recurse in a way that respects gradient
// graph's topological ordering
static void actual_backwards(variable_t* root){
    if(root->grad_meta->num_inputs == 1){
        input_t* input = root->grad_meta->inputs[0];
        update_unary_grad(input, root);
        if(get_ref_count(input->variable) == 0){
            actual_backwards(input->variable);
        } 
    }else if(root->grad_meta->num_inputs == 2){
        input_t* input1 = root->grad_meta->inputs[0];
        input_t* input2 = root->grad_meta->inputs[1];
        update_binary_grads(input1, input2, root);
        if(get_ref_count(input1->variable) == 0){
            actual_backwards(input1->variable);
        }
        if(get_ref_count(input2->variable) == 0){
            actual_backwards(input2->variable);
        }
    }
}

void backwards(variable_t* root){
    NDEBUG_ASSERT(is_scalar(root), "Error: root variable is not a scalar.");
    // set root gradient to 1
    tensor_set_to_scalar_value(root->gradient, 1);
    actual_backwards(root);
}


void set_unary_grad_meta(variable_t* output, variable_t* parent, variable_unary_grad_op_t grad_op){
    input_t* input = input_new(parent, (variable_grad_op_t) grad_op);
    grad_meta_t* new_grad_meta = (grad_meta_t*) malloc(sizeof(grad_meta_t));
    new_grad_meta->ref_count = 0;
    new_grad_meta->num_inputs = 1;
    new_grad_meta->inputs[0] = input;
    output->grad_meta = new_grad_meta;
    increment_ref_count(parent);
}


void set_binary_grad_meta(variable_t* output, variable_t* input1, variable_t* input2, variable_binary_grad_op_t grad_op1, variable_binary_grad_op_t grad_op2){
    input_t* diff_input1 = input_new(input1, (variable_grad_op_t) grad_op1);
    input_t* diff_input2 = input_new(input2, (variable_grad_op_t) grad_op2);
    grad_meta_t* new_grad_meta = (grad_meta_t*) malloc(sizeof(grad_meta_t));
    new_grad_meta->ref_count = 0;
    new_grad_meta->num_inputs = 2;
    new_grad_meta->inputs[0] = diff_input1;
    new_grad_meta->inputs[1] = diff_input2;
    output->grad_meta = new_grad_meta;
    increment_ref_count(input1);
    increment_ref_count(input2);
}
