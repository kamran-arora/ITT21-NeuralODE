import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

def pk_model(t, A, Ka, CL, V):
    """The Pk model"""
    A1, A2 = A
    dA1_dt = -(Ka) * A1
    dA2_dt = (Ka) * A1 - (CL / (V)) * A2
    return [dA1_dt, dA2_dt]

def run_simulation(trajectory_number=10, av_weight=70, Dose=1, F=1, sigma=0.05, t_start=0, t_end=10, dt=0.01):
    # Population size
    trajectory_number = trajectory_number

    # Fixed dose parameters
    av_weight = av_weight
    Dose = Dose   # Administered dose
    F = F      # Bioavailability
    sigma = sigma  # Variability (random noise)

    # Parameter variability and means
    W_ka, B_ka = 0.3, 0.01 * av_weight
    W_cl, B_cl = 0.3, np.log(0.053 * av_weight)
    W_v, B_v = 0.3, 0.733 * av_weight
    B_clwt = 0.2

    # Time settings
    t_start, t_end, dt = t_start, t_end, dt
    T = np.arange(t_start, t_end, dt)

    # Generate inter-individual variability
    Omega = np.diag([W_ka**2, W_cl**2, W_v**2])
    eta_array = np.random.multivariate_normal([0, 0, 0], Omega, 1)
    eta_array = eta_array[0]
    
    Weightlist = np.random.normal(0, 0, 1)  # Individual weights
    Weightlist = Weightlist[0]
    Ka = B_ka * np.exp(eta_array[0])
    CL = np.exp(B_cl + B_clwt * Weightlist + eta_array[0])
    V = B_v * np.exp(eta_array[0])

    # Print the types of Ka, CL, V and shape
    print(f"CL values: {CL}")
    print(f"Ka values: {Ka}")
    print(f"V values: {V}")

    # Storage for solutions
    traj_data = np.full((trajectory_number, len(T)), np.nan)
    numerical_data = np.full((1, len(T)), np.nan)

    # Ensure valid parameters
    denominator = (V * Ka - CL)
    print(f"The denom is {denominator}")
    
    solution = solve_ivp(pk_model, [t_start, t_end], [Dose, 0], t_eval=T, args=(Ka, CL, V))
    print(f"The type of solution is: {type(solution)}")
    numerical_data = solution.y[1]

    for trajectories in range(trajectory_number):
        """Running many random trajectories"""

        eps_list = np.random.normal(0.0, sigma, len(T))  # Noise for each time point
        # Corrected Analytical Solution 
        analytic_solution = (Dose * V * F * Ka / denominator) * (np.exp(-(CL / V) * T) - np.exp(-Ka * T))
        # Add noise correctly
        A2_noisy = analytic_solution * np.exp(eps_list)
        
        traj_data[trajectories, :] = A2_noisy

    # Save data to files
    os.makedirs('data', exist_ok=True)
    np.save('data/traj_data_per_person.npy', traj_data)
    np.save('data/ODE_per_person.npy', numerical_data)

    # Print saved data
    

    # Plot trajectories
    plt.figure(figsize=(8, 5))
    plt.plot(T, numerical_data, linestyle='dashed', label="Numerical", alpha=0.5)
    for i in range(trajectory_number):
        plt.plot(T, traj_data[i, :], label=f"Analytical {i}", alpha=0.5)

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.title("PK Model: Individual Trajectories")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()