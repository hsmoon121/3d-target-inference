import numpy as np
from scipy.interpolate import interp1d


def average_multidim_trajectories(trajectories):
    # Get the longest trajectory to define the common timeline
    max_length = max([len(trajectory) for trajectory in trajectories])

    common_timeline = np.linspace(0, 1, max_length)

    # Determine the number of dimensions from the first trajectory
    num_dimensions = trajectories[0].shape[1]

    # Create a list to hold the interpolated trajectories
    interp_trajectories = []

    for trajectory in trajectories:
        # Normalize the timeline of the current trajectory
        timeline = np.linspace(0, 1, len(trajectory))
        
        # Create a list to hold the interpolated dimensions of the current trajectory
        interp_trajectory = []
        
        # Interpolate each dimension independently
        for dimension in range(num_dimensions):
            # Create an interpolation function for the current dimension of the current trajectory
            interp_func = interp1d(timeline, trajectory[:, dimension], kind='linear', fill_value='extrapolate')
            
            # Interpolate the dimension on the common timeline and add it to our list
            interp_trajectory.append(interp_func(common_timeline))
        
        # Stack the interpolated dimensions to form the interpolated trajectory
        interp_trajectory = np.stack(interp_trajectory, axis=-1)
        
        # Add the interpolated trajectory to our list
        interp_trajectories.append(interp_trajectory)

    # Convert list of trajectories to 3D numpy array
    interp_trajectories = np.array(interp_trajectories)
    
    # Average the trajectories
    average_trajectory = np.mean(interp_trajectories, axis=0)

    return average_trajectory


def average_multidim_velocity_magnitude(trajectories):
    # Get the longest trajectory to define the common timeline
    max_length = max([len(trajectory) for trajectory in trajectories])

    common_timeline = np.linspace(0, 1, max_length)

    # Determine the number of dimensions from the first trajectory
    num_dimensions = trajectories[0].shape[1]

    # Create a list to hold the interpolated velocity magnitudes
    interp_velocity_magnitudes = []

    for trajectory in trajectories:
        # Normalize the timeline of the current trajectory
        timeline = np.linspace(0, 1, len(trajectory))

        # Calculate the velocity for the current trajectory
        velocities = np.diff(trajectory, axis=0)
        velocity_magnitude = np.linalg.norm(velocities, axis=-1)

        # Create an interpolation function for the velocity magnitudes of the current trajectory
        interp_func = interp1d(timeline[:-1], velocity_magnitude, kind='linear', fill_value='extrapolate')
        
        # Interpolate the velocity magnitude on the common timeline and add it to our list
        interp_velocity_magnitudes.append(interp_func(common_timeline[:-1]))

    # Convert list of velocity magnitudes to 2D numpy array
    interp_velocity_magnitudes = np.array(interp_velocity_magnitudes)
    
    # Average the velocity magnitudes
    average_velocity_magnitude = np.mean(interp_velocity_magnitudes, axis=0)

    return average_velocity_magnitude