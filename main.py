import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def pk_model(t, A, Ka, CL, V):
    A1, A2 = A
    dA1_dt = -(Ka) * A1
    dA2_dt = (Ka) * A1 - (CL / (V)) * A2
    return [dA1_dt, dA2_dt]



def run_simulation(N=1, av_weight=70, Dose=1, F=1, sigma=0.05, t_start=0, t_end=10, dt=0.01):
    # Population size
    N = N

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
    eta_array = np.random.multivariate_normal([0, 0, 0], Omega, N)
    
    Weightlist = np.random.normal(0, 0, N)  # Individual weights

    Ka_values = B_ka * np.exp(eta_array[:, 0])
    Ke_values = B_ka * np.exp(eta_array[:, 0])
    CL_values = np.exp(B_cl + B_clwt * Weightlist + eta_array[:, 1])
    V_values = B_v * np.exp(eta_array[:, 2])

    print(f"CL values: {CL_values}")
    print(f"Ka values: {Ka_values}")
    print(f"V values: {V_values}")

    # Storage for solutions
    analytic_data = np.full((N, len(T)), np.nan)
    numerical_data = np.full((N, len(T)), np.nan)

    for i in range(N):
        Ka, CL, V = Ka_values[i], CL_values[i], V_values[i]
        
        # Ensure valid parameters
        denominator = (V * Ka - CL)
        print(f"The denom is {denominator}")
        # if denominator <= 0:
        #     print(f"Skipping individual {i} due to invalid parameters: Ka={Ka}, CL={CL}, V={V}")
        #     continue
        
        # Solve ODE numerically
        solution = solve_ivp(pk_model, [t_start, t_end], [Dose, 0], t_eval=T, args=(Ka, CL, V))
        numerical_data[i, :] = solution.y[1]
        eps_list = np.random.normal(0.0, sigma, len(T))  # Noise for each time point
        # Corrected Analytical Solution 
        A2_analytic = (Dose * V * F * Ka / denominator) * (np.exp(-(CL / V) * T) - np.exp(-Ka * T))
        # Add noise correctly
        A2_noisy = A2_analytic * np.exp(eps_list)
        
        analytic_data[i, :] = A2_noisy

    # Plot trajectories
    plt.figure(figsize=(8, 5))
    for i in range(N):
        plt.plot(T, analytic_data[i, :], label=f"Analytical {i}", alpha=0.5)
        plt.plot(T, numerical_data[i, :], linestyle='dashed', label=f"Numerical {i}", alpha=0.5)

    plt.xlabel("Time (h)")
    plt.ylabel("Concentration")
    plt.title("PK Model: Individual Trajectories")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()