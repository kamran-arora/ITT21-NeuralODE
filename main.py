import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

np.random.seed(10)
# Population size
N = 10 # Single individual for debugging

# Fixed dose parameters
av_weight = 70
Dose = 1   # Administered dose
F = 1      # Bioavailability
sigma = 0.01  # Variability (random noise)

# Parameter variability and means
W_ka, B_ka = 0.3, 0.01*av_weight
W_cl, B_cl = 0.3, np.log(0.053*av_weight)
W_v, B_v = 0.3, 0.733*av_weight
B_clwt = 0.0

# Time settings
t_start, t_end, dt = 0, 10, 0.01
T = np.arange(t_start, t_end, dt)

# Generate inter-individual variability
Omega = np.diag([W_ka**2, W_cl**2, W_v**2])
eta_array = np.random.multivariate_normal([0, 0, 0], Omega, N)
eps_list = np.random.normal(0.0, sigma, len(T))  # Noise for each time point
Weightlist = np.random.normal(0, 0, N)  # Individual weights


Ka_values = B_ka * np.exp(eta_array[:, 0])
Ke_values = B_ka * np.exp(eta_array[:, 0])
CL_values = np.exp(B_cl + B_clwt * Weightlist + eta_array[:, 1])
V_values = B_v * np.exp(eta_array[:, 2])

print(f"CL values: {CL_values}")
print(f"Ka values: {Ka_values}")
print(f"V values: {V_values}")

# Define the ODE system
def pk_model(t, A, Ka, CL, V):
    A1, A2 = A
    dA1_dt = -(Ka) * A1
    dA2_dt = (Ka) * A1 - (CL / (V)) * A2
    return [dA1_dt, dA2_dt]

# Storage for solutions
analytic_data = np.full((N, len(T)), np.nan)
numerical_data = np.full((N, len(T)), np.nan)

for i in range(N):
    Ka, CL, V, Ke = Ka_values[i], CL_values[i], V_values[i], Ke_values
    
    # Ensure valid parameters
    denominator = (V * Ka - CL)
    print(f"The denom is {denominator}")
    # if denominator <= 0:
    #     print(f"Skipping individual {i} due to invalid parameters: Ka={Ka}, CL={CL}, V={V}")
    #     continue
    
    # Solve ODE numerically
    solution = solve_ivp(pk_model, [t_start, t_end], [Dose, 0], t_eval=T, args=(Ka, CL, V))
    numerical_data[i, :] = solution.y[1]
    
    # Corrected Analytical Solution 
    A2_analytic = (Dose * F * Ka / denominator) * (np.exp(-(CL / V) * T)-np.exp(-Ka * T) )
    #A2_analytic = (Dose * F * Ka / denominator) * (1 - np.exp(-(CL / V) * T))
    # Add noise correctly
    A2_noisy = A2_analytic*np.exp(eps_list)
    
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