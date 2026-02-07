from datetime import datetime
import glob
import os
import random
import shutil
from typing import Dict, List
import numpy as np 
from tqdm import tqdm

import torch
from torch import nn

import onnx
import onnxruntime
from onnx import version_converter
from onnxruntime import quantization
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

class SimpleFC(nn.Module):
    def __init__(self):
        super(SimpleFC, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 128)
        self.fc6 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    
class CallibationDataset(CalibrationDataReader):
    """
    A class used to read calibration data for a given model.

    Attributes
    ----------
    calibration_image_folder : str
        The path to the folder containing calibration images
    model_path : str
        The path to the ONNX model file

    Methods
    -------
    get_next() -> Dict[str, List[float]]
        Returns the next item from the enumerator
    rewind() -> None
        Resets the enumeration of calibration data
    """

    def __init__(self, model_path: str) -> None:
        """
        Initializes the ImageNetDataReader class.

        Parameters
        ----------
        model_path : str
            The path to the ONNX model file
        """

        # Use inference session to get input shape
        session = onnxruntime.InferenceSession(model_path, None)
        (_, input_features) = session.get_inputs()[0].shape
        self.input_name = session.get_inputs()[0].name

        # Generate random calibration data
        self.data_list = [np.random.randn(1, input_features).astype(np.float32) * 1 for _ in range(10000)]

        self.enum_data = None  # Initialize enumerator to None


    def get_next(self) -> Dict[str, List[float]]:
        """
        Returns the next item from the enumerator.

        Returns
        -------
        Dict[str, List[float]]
            A dictionary containing the input name and corresponding data
        """

        if self.enum_data is None:
            # Create an iterator that generates input dictionaries
            # with input name and corresponding data
            self.enum_data = iter(
                [{self.input_name: d} for d in self.data_list]
            )
        
        return next(self.enum_data, None)  # Return next item from enumerator

    def rewind(self) -> None:
        """
        Resets the enumeration of calibration data.
        """

        self.enum_data = None  # Reset the enumeration of calibration data


input_model = "./simple_fc.onnx"
infer_model = "./simple_fc_infer.onnx"
quant_model = "./simple_fc_quant.onnx"

fc = SimpleFC()

# Training loop to fit f(x, y) = x^2 + y^2
optimizer = torch.optim.Adam(fc.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 1000
batch_size = 64

for epoch in range(num_epochs):
    # Generate random training data
    inputs = torch.randn(batch_size, 2)
    targets = (inputs[:, 0]**2 + inputs[:, 1]**2).unsqueeze(1)
    
    # Forward pass
    outputs = fc(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

example_inputs = torch.Tensor([[0.5, 0.5]])
print(fc(example_inputs))
# exit()

onnx_program = torch.onnx.export(
    fc,
    example_inputs,
    dynamo=True,
)


onnx_program.save(input_model)
quantization.quant_pre_process(input_model_path=input_model, output_model_path=infer_model, skip_optimization=False)

dr = CallibationDataset("./simple_fc.onnx")

quantize_static(
        infer_model,
        quant_model,
        dr,
        calibrate_method=CalibrationMethod.MinMax, 
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8, 
        activation_type=QuantType.QInt8, 
        reduce_range=True,
        extra_options={'WeightSymmetric': True, 'ActivationSymmetric': False})

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(current_time + ' - ' + '{} model has been created.'.format(os.path.basename(quant_model)))

quantized_session = onnxruntime.InferenceSession(quant_model)
input_name = quantized_session.get_inputs()[0].name
label_name = quantized_session.get_outputs()[0].name
data = example_inputs.numpy()
result = quantized_session.run([label_name], {input_name: data.astype(np.float32)})[0]
print(result)