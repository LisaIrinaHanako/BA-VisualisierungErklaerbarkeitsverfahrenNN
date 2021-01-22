

# Function to shape x-dimensional tensors into 2d tensors    
def reshape(tensor):
    # get originial shape of the tensor
    shape = tensor.shape
    # select and keep first dimension
    first_dim = shape[0]
    # select left dimensions to calculate size of second dimension of the result
    left_dims = len(shape[1:])
    second_dim = 1
    # calculate size of second dimension of the result
    for cur_dim in range(1,left_dims+1):
        second_dim = second_dim * shape[cur_dim]
    # reshape tensor to 2d tensor
    tensor = tensor.reshape(first_dim, second_dim)
    return tensor