import torch
import torch.nn as nn
from torch.autograd import Function

import numpy as np
import qiskit
from qiskit.providers.aer import AerError

from qiskit.visualization import plot_state_qsphere,plot_histogram, plot_state_city
import matplotlib.pyplot as plt

# define global variable
N_INPUTS = 4
N_QUBITS = 1
N_OUT=2
N_SHOTS=512

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
def get_noise(p):
    error_meas = pauli_error([('X',p), ('I', 1 - p)])

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
        
    return noise_model


class QuantumCircuit: 
    """ 
    This class provides a simple interface for interaction  with the quantum circuit 
    """

    def __init__(self, n_qubits, backend, shots, is_add_noise=False, n_inputs=N_INPUTS):
        # -- Circuit definitaion --

        self.n_qubits = n_qubits
        self.n_inputs = n_inputs

        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")

        all_qubits = [i for i in range(n_qubits)]
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)

        self.circuit.measure_all()
        # ------------------------

        self.is_add_noise = is_add_noise
        self.backend = backend
        self.shots = shots

    def forward(self, thetas):
        t_qc = qiskit.transpile(self.circuit, self.backend)
        q_obj = qiskit.assemble(t_qc, shots=self.shots, parameter_binds=[{self.theta: theta.item()} for theta in thetas])

        # run simulator
        noise_model = get_noise(0.01)
        job = self.backend.run(q_obj, noise_model=noise_model if self.is_add_noise else None, shots=self.shots)
        result = job.result().get_counts()
            
        exp = []
        for dict_ in result:
            counts = np.array(list(dict_.values()))
            states = np.array([int(k, 2) for k in list(dict_.keys())])
            prob = counts / self.shots
            expectation = states * prob

            exp.extend(expectation)

        while len(exp) < self.n_inputs * (2 ** self.n_qubits):
            exp.extend([0.00])
        
        return np.asarray(exp).T

    
    def plot(self, thetas):
        self.plot_backend = qiskit.Aer.get_backend("statevector_simulator")
        t_qc = qiskit.transpile(self.circuit, self.plot_backend)
        q_obj = qiskit.assemble(t_qc, shots=self.shots, parameter_binds=[{self.theta: theta.item()} for theta in thetas])

        job = self.plot_backend.run(q_obj)
        result = job.result().get_counts()

        # display show
        # display(plot_state_qsphere(job.result().get_statevector()))
        # display(plot_state_city(job.result().get_statevector(), figsize=[16, 9]))
        # display(self.circuit.draw("mpl"))
        # display(plot_histogram(job.result().get_counts()))


class QuantumLayer(Function):
    @staticmethod
    def forward(ctx, inp):
        if not hasattr(ctx, 'QiskitCirc'):
            try:
                simulator = qiskit.Aer.get_backend('aer_simulator')
                simulator.set_options(device='GPU')
                ctx.QiskitCirc = QuantumCircuit(N_QUBITS, simulator, shots=N_SHOTS, n_inputs=N_INPUTS, is_add_noise=False)
            except AerError as e:
                print("Error GPU Simulator")

        exp_value = ctx.QiskitCirc.forward(inp)
        result = torch.tensor([exp_value])
        ctx.save_for_backward(result, inp)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        grad_output = torch.tensor(grad_output, device=device)

        # gia tri ship nay co the thay doi
        SHIFT = np.pi/2

        forward_tensor, i = ctx.saved_tensors
        input_numbers = i
        
        gradients = torch.Tensor().to(device)
        
        for k in range(N_INPUTS):
            shift_right = input_numbers.detach().clone()
            shift_right[k] = shift_right[k] + SHIFT
            shift_left = input_numbers.detach().clone()
            shift_left[k] = shift_left[k] - SHIFT
            
            expectation_right = torch.tensor([ctx.QiskitCirc.forward(shift_right)], device=device)
            expectation_left  = torch.tensor([ctx.QiskitCirc.forward(shift_left)], device=device)

            gradient = expectation_right - expectation_left
            gradients = torch.cat((gradients, gradient.float()))

        result = torch.tensor(gradients, device=device)

        return (result.float() * grad_output.float()).T


# if __name__ == "__main__":
#     x1 = torch.tensor([0.0624]*N_INPUTS, requires_grad=True)

#     qc = QuantumLayer.apply
#     y1 = qc(x1)
#     print(y1)
#     print('y1 after quantum layer: {}'.format(y1.shape))

#     y1 = nn.Linear(2 ** N_QUBITS * N_INPUTS,1)(y1.float())
#     y1.backward()

#     print('x.grad = {}'.format(x1.grad))