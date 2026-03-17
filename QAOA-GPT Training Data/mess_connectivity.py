from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService()

backend = service.backend("ibm_osaka")  # example backend

coupling_map = backend.configuration().coupling_map

print(coupling_map)