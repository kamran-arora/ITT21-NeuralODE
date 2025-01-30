import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def pk_model(t, A, Ka, CL, V):
    """Pharmacokinetic model."""
    A1, A2 = A
    dA1_dt = -Ka * A1
    dA2_dt = Ka * A1 - (CL / V) * A2
    return [dA1_dt, dA2_dt]

def run_simulation(num_trajectories=10, avg_weight=70, dose=1, bioavailability=1, noise_level=0.05, 
                    time_start=0, time_end=10, time_step=0.4):
    
    print(f"Measurements are taken every {time_step*60} Minutes")
    # Parameter variability and means
    W_ka, B_ka = 0.3, 0.01 * avg_weight
    W_cl, B_cl = 0.3, np.log(0.053 * avg_weight)
    W_v, B_v = 0.3, 0.733 * avg_weight
    B_clwt = 0.2

    # Time settings
    time_points = np.arange(time_start, time_end, time_step)

    # Generate inter-individual variability
    covariance_matrix = np.diag([W_ka**2, W_cl**2, W_v**2])
    variability_factors = np.random.multivariate_normal([0, 0, 0], covariance_matrix, 1)[0]
    
    individual_weight = np.random.normal(0, 0, 1)[0]
    ka = B_ka * np.exp(variability_factors[0])
    cl = np.exp(B_cl + B_clwt * individual_weight + variability_factors[0])
    v = B_v * np.exp(variability_factors[0])

    print(f"CL values: {cl}")
    print(f"Ka values: {ka}")
    print(f"V values: {v}")

    # Storage for solutions
    analytical_solutions = np.full((num_trajectories, len(time_points)), np.nan)
    numerical_solution = np.full((1, len(time_points)), np.nan)

    # Ensure valid parameters
    denominator = (v * ka - cl)
    print(f"Denominator: {denominator}")
    
    solution = solve_ivp(pk_model, [time_start, time_end], [dose, 0], t_eval=time_points, args=(ka, cl, v))
    print(f"Solution type: {type(solution)}")
    numerical_solution = solution.y[1]

    for i in range(num_trajectories):
        noise = np.random.normal(0.0, noise_level, len(time_points))
        
        analytic_solution = (dose * v * bioavailability * ka / denominator) * (np.exp(-(cl / v) * time_points) - np.exp(-ka * time_points))
        noisy_solution = analytic_solution * np.exp(noise)
        
        analytical_solutions[i, :] = noisy_solution

    # Save data to files
    os.makedirs('data', exist_ok=True)
    np.save('data/analytical_solutions.npy', analytical_solutions)
    np.save('data/numerical_solution.npy', numerical_solution)

    # Plot trajectories
    plt.figure(figsize=(8, 5))
    plt.plot(time_points, numerical_solution, linestyle='dashed', alpha=0.5)
    for i in range(num_trajectories):
        plt.plot(time_points, analytical_solutions[i, :], alpha=0.5)

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.title(f"PK Model: {num_trajectories} Unique Trajectories")
    plt.show()

if __name__ == "__main__":
    run_simulation()
