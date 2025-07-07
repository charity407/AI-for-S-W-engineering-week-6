from qiskit import QuantumCircuit
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit.primitives import Estimator
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

# Define the molecule
driver = PySCFDriver(atom='H 0 0 0; H 0 0 0.735', basis='sto-3g')
problem = driver.run()

# Map to qubit Hamiltonian
mapper = JordanWignerMapper()
hamiltonian = mapper.map(problem.hamiltonian)

# Set up the VQE algorithm
ansatz = TwoLocal(hamiltonian.num_qubits, 'ry', 'cz', reps=2, entanglement='full')
optimizer = SPSA(maxiter=100)
estimator = Estimator()
vqe = VQE(estimator, ansatz, optimizer)

# Solve for the ground state energy
result = vqe.compute_minimum_eigenvalue(hamiltonian)
print(f"Ground state energy: {result.eigenvalue.real} Ha")