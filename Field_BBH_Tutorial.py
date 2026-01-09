# coding:utf-8
from Field_modeling import Field_BBH

# print("Generating default MW Field BBH population data...")
#
# # This will run the simulation, resample the data, and save it to ./data/
# # You can customize parameters here if needed
# Field_BBH.simulate_and_save_default_population(
#     n_sim_samples=100000,
#     target_N=50000
# )
#
# print("Setup complete. You can now use Field_BBH functions directly.")
# raise ValueError

# FEATURE 1: Randomly Sample eccentricities and plot CDF
# No object instantiation needed. It loads data automatically.
print("Sampling eccentricities...")
N = 5000
e_samples = Field_BBH.generate_eccentricity_samples(size=N)

print('Example output (eccentricities):', e_samples[:3])
Field_BBH.plot_eccentricity_cdf(e_samples, label="Sampled Field BBH")

# FEATURE 2: Generate Snapshot and Plot
print("\nGenerating MW Field BBH Snapshot...")
snapshot = Field_BBH.generate_snapshot(t_window_Gyr=10.0, num_realizations=20)

print(f"Snapshot generated with {len(snapshot)} systems.")
Field_BBH.plot_snapshot(snapshot)