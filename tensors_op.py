import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

#print("torch version : {}".format(torch.__version__))

digit_0_array_og = cv2.imread("mnist_0.jpg")
digit_1_array_og = cv2.imread("mnist_1.jpg")

digit_0_array_gray = cv2.imread("mnist_0.jpg", cv2.IMREAD_GRAYSCALE)
digit_1_array_gray = cv2.imread("mnist_1.jpg", cv2.IMREAD_GRAYSCALE)

'''
fig, axs = plt.subplots(1,2, figsize=(10,5))

axs[0].imshow(digit_0_array_og, cmap='gray',interpolation='none')
axs[0].set_title("Digit 0 Image")
axs[0].axis('off')

axs[1].imshow(digit_1_array_og, cmap="gray", interpolation = 'none')
axs[1].set_title("Digit 1 Image")
axs[1].axis('off')

plt.show()
print(digit_0_array_og.shape)
print(digit_0_array_gray.shape)
'''


'''

img_tensor_0 = torch.tensor(digit_0_array_og, dtype=torch.float32) /255.0
img_tensor_1 = torch.tensor(digit_1_array_og, dtype=torch.float32) / 255.0

#print(img_tensor_0.shape)


#creating input batch
batch_tensor = torch.stack([img_tensor_0, img_tensor_1])
#print(batch_tensor.shape)

# current tensor shape is B,H,W,C , pytorch expects B,C,H,W.  so use torch.permute() 
batch_input = batch_tensor.permute(0,3,1,2)
#print(batch_input.shape)

# creating custom tensors
ones = torch.ones(5)
print(ones)

b = torch.zeros(2, 3)
print(b)

c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(c)
# and so on

#accessing element of a particular index
print(c[2])

print(b[1, 2])

print(b[:]) # all elements
print(c[1:3]) # all elements from index 1 to 2

print(c[:4]) # all elements till index 4(not including 4)
print(b[0,:]) # first row and all columns
print(b[:,1]) # all rows and second column

# converting tensor types
float_tensor = torch.tensor([[1, 2, 3],[4., 5, 6]])
int_tensor = float_tensor.type(torch.int64)
print(float_tensor.dtype)
print(int_tensor.dtype)
print(int_tensor)

# tensor to/from numpy

# tensor to numpy
c_numpy = c.numpy()
print(c_numpy)

# numpy to tensor
k = np.ones(5)
print(k)
k_tensor = torch.from_numpy(k)
print(k_tensor)
'''
#Arithmetic operations
'''

tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 = torch.tensor([[-1,2,-3],[4,-5,6]])

#addition
print(tensor1+tensor2)
print(torch.add(tensor1,tensor2))

# Subtraction
print(tensor1-tensor2)
print(torch.sub(tensor1,tensor2))

# Multiplication
# Tensor with Scalar
print(tensor1 * 2)

# Tensor with another tensor
# Elementwise Multiplication
print(tensor1 * tensor2)

# Matrix multiplication
tensor3 = torch.tensor([[1,2],[3,4],[5,6]])
print(torch.mm(tensor1,tensor3))

# Division
# Tensor with scalar
print(tensor1/2)

# Tensor with another tensor
# Elementwise division
print(tensor1/tensor2)
'''
# broadcasting
'''
a is a 1-dimensional tensor with shape ([ 3 ]).
b is a scalar tensor with shape ([ 1 ]).
When adding a and b, PyTorch broadcasts b to match the shape of a, resulting in ([ 1 + 4, 2 + 4, 3 + 4 ]).'''

a = torch.tensor([1, 2, 3])
b = torch.tensor([4])

result = a + b
print("Result of Broadcasting:\n",result)

'''
a is a 2-dimensional tensor with shape ([1, 3]).

b is a 2-dimensional tensor with shape ([3, 1]).

When adding a and b, PyTorch broadcasts both tensors to the common shape ([3, 3]), resulting in:

[  1+4 2+4 3+4
   1+5 2+5 3+5
   1+6 2+6 3+6].
'''

a = torch.tensor([[1, 2, 3]])
b = torch.tensor([[4], [5], [6]])

# adding tensors of different shapes
result = a + b
print("Shape: ", result.shape)
print("\n")
print("Result of Broadcasting:\n", result)

# cpu and gpu
tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cpu')
tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')

#moving tensors from cpu to gpu and vice versa
tensor_gpu2cpu = tensor_gpu.to(device='cpu')
tensor_cpu2gpu = tensor_cpu.to(device='cuda')

