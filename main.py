import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def pk_model(t, A, Ka, CL, V):
    """This is the pk model ODE system"""
    A1, A2 = A
    dA1_dt = -Ka * A1
    dA2_dt = Ka * A1 - (CL / V) * A2
    return [dA1_dt, dA2_dt]

def run_simulation(population_size=10, avg_weight=70, dose=1, bioavailability=1, noise_level=0.01, 
                    time_start=0, time_end=10, time_step=0.4):
    """We run the simulation for a certain population size. Each person will be slightly different, with individual varied trajectories generated for each person"""
    print(f"Measurements are taken every {time_step*60} Minutes")
    print(f"we are measuring the responese for {population_size} people")
    # Parameter variability and means
    W_ka, B_ka = 0.3, 0.01 * avg_weight 
    W_cl, B_cl = 0.3, np.log(0.053 * avg_weight)
    W_v, B_v = 0.3, 0.733 * avg_weight
    B_clwt = 0.2

    # Time settings
    time_points = np.arange(time_start, time_end, time_step) 

    # Generate inter-individual variability
    covariance_matrix = np.diag([W_ka**2, W_cl**2, W_v**2])
    variability_factors = np.random.multivariate_normal([0, 0, 0], covariance_matrix, population_size)
    
    individual_weights = np.random.normal(0, 0, population_size)  # Individual weights

    ka_values = B_ka * np.exp(variability_factors[:, 0])
    cl_values = np.exp(B_cl + B_clwt * individual_weights + variability_factors[:, 1])
    v_values = B_v * np.exp(variability_factors[:, 2])

    print(f"CL values: {cl_values}")
    print(f"Ka values: {ka_values}")
    print(f"V values: {v_values}")

    # Storage for solutions
    analytical_solutions = np.full((population_size, len(time_points)), np.nan)
    numerical_solutions = np.full((population_size, len(time_points)), np.nan)

    for i in range(population_size):
        ka, cl, v = ka_values[i], cl_values[i], v_values[i]
        
        # Ensure valid parameters
        denominator = (v * ka - cl)
        # print(f"Denominator: {denominator}")
        
        # Solve ODE numerically
        solution = solve_ivp(pk_model, [time_start, time_end], [dose, 0], t_eval=time_points, args=(ka, cl, v))
        numerical_solutions[i, :] = solution.y[1]

        # Generate noise
        noise = np.random.normal(0.0, noise_level, len(time_points))
        
        # Corrected analytical solution 
        A2_analytic = (dose * v * bioavailability * ka / denominator) * (np.exp(-(cl / v) * time_points) - np.exp(-ka * time_points))
        A2_noisy = A2_analytic * np.exp(noise)
        
        analytical_solutions[i, :] = A2_noisy

    # Save data to files
    os.makedirs('data', exist_ok=True)
    np.save('data/analytical_solutions.npy', analytical_solutions)
    np.save('data/numerical_solutions.npy', numerical_solutions)

    # Plot trajectories
    plt.figure(figsize=(8, 5))
    for i in range(population_size):
        plt.plot(time_points, analytical_solutions[i, :], alpha=0.5)
        plt.plot(time_points, numerical_solutions[i, :], linestyle='dashed', alpha=0.5)

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.title("PK Model: Individual Trajectories")
    plt.show()

if __name__ == "__main__":
    run_simulation()
