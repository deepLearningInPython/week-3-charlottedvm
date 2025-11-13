import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array, kernel_array):
    return len(input_array) - len(kernel_array) + 1

# -----------------------------------------------


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array, kernel_array):
    # length convolution output
    conv_output_len = compute_output_size_1d(input_array, kernel_array)
    # initializing an empty output array
    output_array = np.zeros(conv_output_len)
    # Fill the cells in the array with a loop.
    for i in range(conv_output_len):
        output_array[i] = np.sum(input_array[i:i+len(kernel_array)] * kernel_array)

    return output_array

# -----------------------------------------------


# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_2d(input_matrix, kernel_matrix):
    return input_matrix.shape[0] - kernel_matrix.shape[0] + 1, input_matrix.shape[1] - kernel_matrix.shape[1] + 1

# -----------------------------------------------

# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
# -----------------------------------------------

def convolute_2d(input_matrix, kernel_matrix):
   # length convolution output
   output_array = np.zeros(compute_output_size_2d(input_matrix, kernel_matrix))
   print(output_array.shape[0], output_array.shape[1], output_array )

   # define which parts of the input matrix need to be multiplied with the kernel matrix.
   for i in range(output_array.shape[0]):
    for j in range(output_array.shape[1]):
       output_array[i,j] = np.sum(input_matrix[i:i+kernel_matrix.shape[0], j:j+kernel_matrix.shape[1]] * kernel_matrix)

   return output_array

# -----------------------------------------------