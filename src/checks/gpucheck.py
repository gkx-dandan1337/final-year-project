import torch

print(torch.cuda.is_available())  # True if GPU is available
print(torch.cuda.current_device())  # Index of current GPU
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # GPU name
