import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value in self.cache
        self.cache = dict()
    
    def activation_f(self, func_name, input):
        """
        Applies an activation function in the forward pass.

        Args:
            input: input to the activation function
        """
        output = torch.tensor([])
        if func_name == "relu":
            output = torch.max(input, torch.zeros_like(input))
        elif func_name == "sigmoid":
            output = torch.sigmoid(input)
        elif func_name == "identity":
            output = input
        return output
    
    def activation_b(self, func_name, input):
        """
        Returns the derivative of post-activation w.r.t. pre-activation, in the backward pass.

        Args:
            func_name: activation function to apply
            input: input to the activation function
        """
        output = torch.tensor([])
        if func_name == "relu":
            diagonal_elems = torch.tensor([int(x > 0) for x in input])
            output = torch.diag(diagonal_elems)
        elif func_name == "sigmoid":
            diagonal_elems = torch.sigmoid(input)*(1 - torch.sigmoid(input))
            output = torch.diag(diagonal_elems)
        elif func_name == "identity":
            output = torch.eye(input.size(dim=0))
        return output

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """

        W1, b1 = self.parameters["W1"], self.parameters["b1"]
        W2, b2 = self.parameters["W2"], self.parameters["b2"]

        self.cache['input'] = x

        output = (x @ W1.T) + b1
        # print(f'Computed s1: {output.size()}')
        self.cache['s1'] = output

        output = self.activation_f(self.f_function, output)
        # print(f'Computed a1: {output.size()}')
        self.cache['a1'] = output

        output = (output @ W2.T) + b2
        # print(f'Computed s2: {output.size()}')
        self.cache['s2'] = output

        output = self.activation_f(self.g_function, output)
        # print(f'Computed y_pred: {output.size()}')
        self.cache['y_pred'] = output

        return output
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """

        self.grads['dJdy_hat'] = dJdy_hat
        input = self.cache['input']

        gradients = {'dJdW1':[],
                     'dJdb1':[],
                     'dJdW2':[],
                     'dJdb2':[]}

        # Compute derivatives for each input in batch separately
        for i in range(len(input)):
            dy_hatds2 = self.activation_b(self.g_function, self.cache['s2'][i])
            ds2da1 = self.parameters['W2']
            da1ds1 = self.activation_b(self.f_function, self.cache['s1'][i])

            dJdb2i = dJdy_hat[i].unsqueeze(0) @ dy_hatds2.float()
            gradients['dJdb2'].append(dJdb2i)

            dJdW2i = dJdb2i.T @ self.cache['a1'][i].unsqueeze(0)
            gradients['dJdW2'].append(dJdW2i)

            dJdb1i = dJdb2i @ ds2da1
            dJdb1i = dJdb1i @ da1ds1.float()
            gradients['dJdb1'].append(dJdb1i)

            dJdW1i = dJdb1i.T @ input[i].unsqueeze(0)
            gradients['dJdW1'].append(dJdW1i)

        # Average per-input derivatives to get derivatives for the whole batch
        for param in gradients.keys():
            stacked = torch.stack(gradients[param], dim=0)
            summed = torch.sum(stacked, dim=0)
            self.grads[param] = summed/len(gradients[param])      
    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # compute loss and derivative w.r.t. y_hat
    J = torch.mean((y - y_hat)**2)
    dJdy_hat = 2 * (y_hat - y) / y.shape[1]

    # return loss, dJdy_hat
    return J, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # add small positive constant to avoid div by zero
    ep = 1e-32

    # clamp y_hat values between -100 and 100
    J = -torch.mean((y*torch.clamp(torch.log(y_hat), -100, 100)) + (1-y)*torch.clamp(torch.log(1-y_hat), -100, 100))
    dJdy_hat = torch.div((-torch.div(y, y_hat+ep)) + torch.div(1-y, 1-y_hat+ep), y.size(dim=1))

    # return loss, dJdy_hat
    return J, dJdy_hat
    