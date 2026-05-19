import torch
import matplotlib.pyplot as plt



x = torch.tensor([2.0, 5.0], requires_grad= True)
y = torch.tensor([3.0, 7.0], requires_grad= True)

z = x*y + y**2
z.retain_grad() #By default intermediate layer weight updation is not shown.

#compute the gradient
z_sum = z.sum().backward()


print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
print(f"Gradient of z: {z.grad}")
print(f"Result of the operation: z = {z.detach()}")


