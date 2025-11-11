import onnxruntime as ort
from torch.utils.data import DataLoader
import numpy as np
from numpy.typing import NDArray

def create_onnx_session(model_path: str, use_gpu: bool = False) -> ort.InferenceSession:
    """ Create an ONNX Runtime inference session.
    Args:
        model_path (str): Path to the ONNX model file.
        use_gpu (bool): Whether to use GPU for inference.
    Returns:
        ort.InferenceSession: The created ONNX Runtime inference session.
    """
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    return session

def get_model_input_details(session: ort.InferenceSession) -> tuple[str, int | None, int, int, int]:
    """
    Extracts input name and shape details from an ONNX Runtime session.

    Returns:
        input_name (str): The name of the model input node.
        batch_dim (int or None): Batch size if fixed, else None.
        C (int): Number of channels.
        H (int): Input height.
        W (int): Input width.
    """
    input_meta = session.get_inputs()[0] 
    input_name = input_meta.name
    input_shape = input_meta.shape 

    resolved_shape = []
    for dim in input_shape:
        if isinstance(dim, int):
            resolved_shape.append(dim)
        else:
            resolved_shape.append(1)  

    if len(resolved_shape) != 4:
        raise ValueError(f"Unexpected input shape {resolved_shape}. Expected [N, C, H, W].")

    batch_dim, C, H, W = resolved_shape
    dynamic_batch = None if not isinstance(input_shape[0], int) else input_shape[0]

    return input_name, dynamic_batch, C, H, W


def run_onnx_inference(session: ort.InferenceSession, batch: NDArray[np.float32], input_name: str = None) -> NDArray[np.float32]:
    """
    Run inference on a single batch and return model outputs as a NumPy array.
    Args:
        session (ort.InferenceSession): The ONNX Runtime inference session.
        batch (NDArray[np.float32]): Input batch data.
        input_name (str, optional): Name of the model input node. If None, uses the first input.
    Returns:
        NDArray[np.float32]: Model outputs.
    """
    if input_name is None:
        input_name = session.get_inputs()[0].name

    if hasattr(batch, "numpy"):
        batch_np = batch.numpy().astype(np.float32)
    else:
        batch_np = np.ascontiguousarray(batch, dtype=np.float32)

    outputs = session.run(None, {input_name: batch_np})
    return outputs[0]  # Assuming single output
