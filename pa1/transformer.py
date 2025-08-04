import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def transformer(X: ad.Node, nodes: List[ad.Node], 
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, input_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """
    Wq, Wk, Wv, Wo, W1, W2, b1, b2 = nodes

    # Self-attention mechanism
    # Q, K, V projections: (batch, seq_len, input_dim) @ (input_dim, model_dim) -> (batch, seq_len, model_dim)
    Q = X @ Wq  # (batch, seq_len, model_dim)
    K = X @ Wk  # (batch, seq_len, model_dim) 
    V = X @ Wv  # (batch, seq_len, model_dim)
    
    # Scaled dot-product attention
    # QK^T: (batch, seq_len, model_dim) @ (batch, model_dim, seq_len) -> (batch, seq_len, seq_len)
    K_transposed = ad.transpose(K, 1, 2)  # (batch, model_dim, seq_len)
    attention_scores = Q @ K_transposed  # (batch, seq_len, seq_len)
    
    # Scale by sqrt(model_dim)  
    scale_factor = 1.0 / (model_dim ** 0.5)
    scaled_scores = ad.mul_by_const(attention_scores, scale_factor)
    
    # Apply softmax to get attention weights
    attention_weights = ad.softmax(scaled_scores, dim=-1)  # (batch, seq_len, seq_len)
    
    # Apply attention to values
    # (batch, seq_len, seq_len) @ (batch, seq_len, model_dim) -> (batch, seq_len, model_dim)
    attention_output = attention_weights @ V
    
    # Output projection
    # (batch, seq_len, model_dim) @ (model_dim, model_dim) -> (batch, seq_len, model_dim)
    projected_output = attention_output @ Wo
    
    # Layer normalization after attention (Pre-LN style)
    # Normalize over the last dimension (model_dim)
    normed_attention = ad.layernorm(projected_output, normalized_shape=[model_dim], eps=eps)
    
    # Feed-forward layer
    # (batch, seq_len, model_dim) @ (model_dim, model_dim) -> (batch, seq_len, model_dim)
    ff_intermediate = normed_attention @ W1 + b1
    ff_activated = ad.relu(ff_intermediate)  # (batch, seq_len, model_dim)
    ff_output = ff_activated @ W2 + b2  # (batch, seq_len, num_classes)
    
    # Layer normalization after feed-forward
    # Since ff_output has shape (batch, seq_len, num_classes), normalize over num_classes
    normed_ff = ad.layernorm(ff_output, normalized_shape=[num_classes], eps=eps)
    
    # Global average pooling for classificatioj
    output = ad.mean(normed_ff, dim=(1,), keepdim=False)  # (batch, num_classes)
    
    return output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, 1, num_classes) or (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    # Cross-entropy loss
    softmax_probs = ad.softmax(Z)
    log_probs = ad.log(softmax_probs)
    neg_y = ad.mul_by_const(y_one_hot, -1.0)
    # Z has shape (batch, num_classes), y_one_hot has shape (batch, num_classes)
    elementwise_loss = neg_y * log_probs
    # Mean over all dimensions to get scalar loss  
    loss = ad.mean(elementwise_loss, dim=(0, 1), keepdim=False)
    return loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    """TODO: Your code here"""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx >= num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len, :]
        y_batch = y[start_idx:end_idx]
        
        # Ensure X_batch has the right shape for transformer input
        actual_batch_size = X_batch.shape[0]
        
        # Compute forward and backward passes
        X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32).clone().detach()
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).clone().detach()
        # Call f_run_model which expects X_batch, y_batch, model_weights
        result = f_run_model(X_batch_tensor, y_batch_tensor, model_weights)
        
        logits = result[0]
        batch_loss = result[1]
        grads_list = result[2:]  # gradients for W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2
        
        # Update weights and biases
        for i in range(len(model_weights)):
            grad_tensor = torch.tensor(grads_list[i], dtype=torch.float32)
            
            # Handle batch gradients: if gradient has batch dimension, sum over it
            if model_weights[i].shape != grad_tensor.shape:
                if i >= 6:  # Bias terms (b_1 and b_2) need special handling
                    # For bias, sum over batch and sequence dimensions
                    if grad_tensor.dim() == 3:  # [batch, seq, dim]
                        grad_tensor = grad_tensor.sum(dim=(0, 1))  # Sum to [dim]
                    elif grad_tensor.dim() == 2:  # [batch, dim]
                        grad_tensor = grad_tensor.sum(dim=0)  # Sum to [dim]
                elif grad_tensor.dim() == model_weights[i].dim() + 1:
                    # Weight matrices have extra batch dimension, average over it
                    grad_tensor = grad_tensor.mean(dim=0)
                else:
                    print(f"Shape mismatch! Weight {i}: {model_weights[i].shape}, Grad: {grad_tensor.shape}")
                    continue  # Skip this update to avoid shape issues
                    
            model_weights[i] = model_weights[i] - lr * grad_tensor

        # Accumulate the loss
        total_loss += float(batch_loss) * X_batch.shape[0]


    # Compute the average loss
    
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    # TODO: Your code here
    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 64 #
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 100  # Increased back up
    batch_size = 32
    lr = 0.01  # Lower learning rate to avoid NaN

    # TODO: Define the forward graph.
    X = ad.Variable(name="X")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K") 
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")
    
    nodes = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]
    y_predict = transformer(X, nodes, model_dim, seq_length, eps, batch_size, num_classes)
    y_groundtruth = ad.Variable(name="y")
    # Use actual batch size from X shape instead of fixed batch_size
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # TODO: Construct the backward graph.
    grads = ad.gradients(loss, nodes)
    

    # TODO: Create the evaluator.
    # grads: List[ad.Node] = ... # TODO: Define the gradient nodes here
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(model_dim)  # Better initialization scale
    # Attention weights: input_dim -> model_dim
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    # Output projection: model_dim -> model_dim
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    # Feed-forward weights: model_dim -> model_dim -> num_classes
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    # Biases
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(X_batch, y_batch, model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        
        result = evaluator.run(
            input_values={
                X: X_batch,
                y_groundtruth: y_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1], 
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7]
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx >= num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len, :]
            logits = test_evaluator.run({
                X: torch.tensor(X_batch, dtype=torch.float32),
                W_Q: model_weights[0],
                W_K: model_weights[1], 
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7]
            })
            all_logits.append(logits[0].detach().numpy())
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val, dtype=torch.float32, requires_grad=False),
        torch.tensor(W_K_val, dtype=torch.float32, requires_grad=False),
        torch.tensor(W_V_val, dtype=torch.float32, requires_grad=False),
        torch.tensor(W_O_val, dtype=torch.float32, requires_grad=False),
        torch.tensor(W_1_val, dtype=torch.float32, requires_grad=False),
        torch.tensor(W_2_val, dtype=torch.float32, requires_grad=False),
        torch.tensor(b_1_val, dtype=torch.float32, requires_grad=False),
        torch.tensor(b_2_val, dtype=torch.float32, requires_grad=False)
    ]
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
