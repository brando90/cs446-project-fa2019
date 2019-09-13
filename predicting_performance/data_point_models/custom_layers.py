import torch.nn as nn

#mini class to add a flatten layer to the ordered dictionary
class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out # (batch_size, *size)

class View(nn.Module):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, input, shape):
        '''
        TODO: the first dimension is the data batch_size
        so we need to decide how the input shape should be like
        '''
        return input.view(*self.shape)

    def __repr__(self):
        return f'View({shape})'
