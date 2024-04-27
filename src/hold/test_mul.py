import numpy as np
import numpy_multiply as mult_example
import time

# Create two NumPy arrays
# cpu arrays
arr1 = np.array([[1.0, 2.0,3.0, 4.0],[5.0, 6.0,7.0, 8.0]])
arr2 = np.array([[1.0, 2.0,3.0, 4.0],[5.0, 6.0,7.0, 8.0]])
arr3 = np.array([[0,0,0,0],[0,0,0,0]])

# move to gpu
# ptr1 = move_to_gpu(arr1)

# ptr 1 is no longer the same as arr1, operations should use that
# any time i want to see the contents of ptr1 i have to move it back to cpu
conv = lambda x, h: np.apply_along_axis(
    lambda x: np.convolve(x, h.flatten(), mode="full"), axis=0, arr=x
)
def get_vec_filter(f):
    return f[:, np.newaxis]
deriv_f = get_vec_filter(np.array([-1, 1]))

def arrprint(arr1,cols, rows):
  for i in range(rows):
    for j in range(cols):
      # arr1[i * cols + j] = i * cols + j
      print(arr1[i * cols + j],end='\t')
    print("")

flt = np.array([-1,1])
cols = 640
rows = 400
arr1 = np.zeros(rows*cols)

arr2 = np.zeros(rows*cols)
arr3 = np.zeros(rows * cols)
# print(arr1)
for i in range(rows):
  for j in range(cols):
    arr1[i * cols + j] = i * cols + j

arr1x = arr1.reshape(rows, cols)
# print(arr1)
# print(arr1x)
# time.start()
# s1 = time.perf_counter()
s1 = time.perf_counter()
for i in range(100):
  val = conv(arr1x, deriv_f)
  
s2 = time.perf_counter()
print(f"CPU CONV: {s2 - s1}")
# print(val)
    # arr1[i * rows + j] = i*rows+j
# for i in rang
val0 = mult_example.warmup(1)
# mult_example.move_to_gpu_addr(val1, arr1)
val1 = mult_example.move_to_gpu(arr1)
val2 = mult_example.move_to_gpu(arr2)
val3 = mult_example.move_to_gpu(arr3)
val4 = mult_example.move_to_gpu(flt)
s1x = time.perf_counter()
for i in range(100):
  mult_example.direct_conv_wrap(val1, val2,val4, flt.shape[0], cols, rows,arr3.shape[0])
s2x = time.perf_counter()
print(f"GPU CONV: {s2x - s1x}")
print((s2 - s1)/(s2x - s1x))
# mult_example.move_to_cpu(val2, arr3)
# arrprint(arr3, cols, rows)
# mult_example.free_gpu(val1)
# mult_example.free_gpu(val2)
# mult_example.free_gpu(val4)

# flt = np.array([1,1,1,])
# for (int i = 0; i < row; i++) {
#   for (int j = 0; j < col; j++) {
#     printf("%.2f\t", A[i*col+j]);
#   }
#   printf("\n");
# }
# print(arr1)
# cuda_conv(arr1, arr2, flt)
# print(arr1)
# # arr2 = np.array([5.0, 6.0,7.0, 8.0])

# # # Set the length of the array
# # length = 20000

# # # Generate a random NumPy array of specified length
# # arr1 = np.random.rand(length, length)

# # arr2 = np.random.rand(length)
# # Call the C++ function to perform array multiplication

# print(val)

# s1 = time.perf_counter()
# print(arr1)

# mult_example.array_multiply(arr1, arr2)
# s2 = time.perf_counter()
# # print(s2 - s1)
# # result = arr1 * arr2
# # s3 = time.perf_counter()
# print(mult_example.a_number())
# print(f"time to run first {s2 - s1}")#\ntime to run secn {s3 - s2}")
# print(arr1)
# # result2

# # Print the result
# # print("Result of array multiplication:")
# # print(result)
