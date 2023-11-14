import numpy as np


def generate_qap_problem_instance(n):
    """Generates a problem instance for the Quadratic Assignment Problem (QAP).

    Args:
        n: The problem size.

    Returns:
        A tuple of four NumPy arrays:
            * locations: A 2D array of locations.
            * facilities: A 2D array of facilities.
            * distance_matrix: A 2D array of distances between locations.
            * flow_matrix: A 2D array of flows between facilities.
    """

    # Generate random locations.
    locations = np.random.randn(n, 2)

    # Generate random facilities.
    facilities = np.random.randn(n, 2)

    # Generate a distance matrix.
    distance_matrix = np.zeros((n, n))
    for i in range(n):
    	for j in range(n):
        	distance_matrix[i, j] = np.linalg.norm(locations[i] - locations[j])

    # Generate a flow matrix.
    flow_matrix = np.random.rand(n, n)
	flow_matrix = flow_matrix + flow_matrix.T

    return locations, facilities, distance_matrix, flow_matrix


def generate_batch_qap_problem_instance(batch_size, n):
	"""Generates a batch of problem instances for the Quadratic Assignment Problem (QAP).

	Args:
		batch_size: The batch size.
		n: The problem size.

	Returns:
		A tuple of four NumPy arrays:
			* locations: A 3D array of locations.
			* facilities: A 3D array of facilities.
			* distance_matrix: A 3D array of distances between locations.
			* flow_matrix: A 3D array of flows between facilities.
	"""

	locations = np.zeros((batch_size, n, 2))
	facilities = np.zeros((batch_size, n, 2))
	distance_matrix = np.zeros((batch_size, n, n))
	flow_matrix = np.zeros((batch_size, n, n))

	for i in range(batch_size):
		locs, facs, dist_mat, flow_mat = generate_qap_problem_instance(n)
		locations[i], facilities[i], distance_matrix[i], flow_matrix[i] = locs, facs, dist_mat, flow_mat
	locations, facilities, distance_matrix, flow_matrix = map(torch.tensor, (locations, facilities, distance_matrix, flow_matrix))
	return locations, facilities, distance_matrix, flow_matrix
	