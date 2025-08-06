from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        res_mm = input_values[0] @ input_values[1]
        mean = torch.mean(res_mm, dim=(-1), keepdim=True)
        var = torch.var(res_mm, dim=(-1), keepdim=True,correction=0)
        return (res_mm - mean) / torch.sqrt(var + node.attrs["eps"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        input_A, input_B = node.inputs
        res_mm = input_A @ input_B
        res = layernorm(
            res_mm, 
            normalized_shape=node.attrs["normalized_shape"], 
            eps=node.attrs["eps"]
        )
        grad_C = layernorm.gradient(res, output_grad)[0]
        grad_A, grad_B = matmul.gradient(res_mm, grad_C)
        return [grad_A, grad_B]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        res_mm = input_values[0] @ input_values[1]
        max_num = torch.max(res_mm, dim=node.attrs["dim"], keepdim=True).values
        softmax_result = torch.nn.functional.softmax(
            res_mm - max_num, dim=node.attrs["dim"]
        )
        return softmax_result

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        input_A, input_B = node.inputs
        res_mm = input_A @ input_B
        res_softmax = softmax(res_mm, dim=node.attrs["dim"])
        grad_softmax = softmax.gradient(res_softmax, output_grad)[0]
        grad_A, grad_B = matmul.gradient(res_mm, grad_softmax)
        return [grad_A, grad_B]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
