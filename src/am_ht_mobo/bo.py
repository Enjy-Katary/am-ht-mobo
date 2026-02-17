import numpy as np
import polars as pl
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
from botorch.utils import standardize
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from .energy import energy_consumption
import os
from .filemanager import get_file_manager


def load_config():
    """Load configuration from JSON file and return config data and paths."""

    fm = get_file_manager()
    config = fm.config
    paths = fm.get_paths()
    FIGURES_PATH = paths['figures']
    DATABASE_PATH = paths['database_demo']  
    CSV_PATH = paths['pareto_optimal']
    BOUNDS = fm.get_bounds()
    REFERENCE = fm.get_reference_point()
    
    return config, FIGURES_PATH, DATABASE_PATH, CSV_PATH, BOUNDS, REFERENCE

config, FIGURES_PATH, DATABASE_PATH, CSV_PATH, BOUNDS, REFERENCE = load_config()


def load_data():
    """Load predefined SEED points from JSON configuration."""

    T1_values = config['initial_process_parameters']['T1 (C)']
    t1_values = config['initial_process_parameters']['t1 (min)']
    T2_values = config['initial_process_parameters']['T2 (C)']
    t2_values = config['initial_process_parameters']['t2 (min)']
    
    points = np.array([[T1_values[i], t1_values[i], T2_values[i], t2_values[i]] 
                       for i in range(len(T1_values))])
    
    UTS_values = config['initial_process_results']['UTS']
    Energy_values = config['initial_process_results']['Energy']
    
    results = np.array([[UTS_values[i], Energy_values[i]] 
                        for i in range(len(UTS_values))])

    results_transformed = np.copy(results)
    results_transformed[:, 1] = 1.0 / results[:, 1]

    return points, results_transformed, results

def load_experiments(database_path):
    """Load all experiments from the database CSV."""
    
    df = pl.read_csv(database_path, schema_overrides={
        "T1 (C)": pl.Float64,
        "t1 (min)": pl.Float64,
        "T2 (C)": pl.Float64,
        "t2 (min)": pl.Float64,
        "Ultimate Tensile Strength (MPa)": pl.Float64,
        "Energy (kWh)": pl.Float64
    })
    
    if len(df) == 0:
        return None, None, None
    
    n_initial = len(config['initial_process_parameters']['T1 (C)'])
    
    if len(df) < n_initial:
        return None, None, None
    
    train_x = df.select(["T1 (C)", "t1 (min)", "T2 (C)", "t2 (min)"]).to_numpy()
    train_y_original = df.select(["Ultimate Tensile Strength (MPa)", "Energy (kWh)"]).to_numpy()
    
    train_y_transformed = np.copy(train_y_original)
    train_y_transformed[:, 1] = 1.0 / train_y_original[:, 1]
    
    print(f"Loaded {len(train_x)} previous experiments from database")
    
    return train_x, train_y_transformed, train_y_original


def initialize_database(database_path, initial_data):
    """Initialize database with initial experiments."""
    train_x, _, train_y_original = initial_data
    
    initial_df = pl.DataFrame({
        "T1 (C)": train_x[:, 0],
        "t1 (min)": train_x[:, 1],
        "T2 (C)": train_x[:, 2],
        "t2 (min)": train_x[:, 3],
        "Ultimate Tensile Strength (MPa)": train_y_original[:, 0],
        "Energy (kWh)": train_y_original[:, 1]
    })
    
    initial_df.write_csv(database_path)


def initialize_GP_model(train_x, train_y):
    """Initialize and train Gaussian Process surrogate models for multi-objective optimization."""
    train_x = torch.tensor(train_x, dtype=torch.float64)
    train_y = torch.tensor(train_y, dtype=torch.float64)

    # Scale design space bounds to unit cube
    bounds = torch.tensor(BOUNDS, dtype=torch.float64)
    train_x_scaled = (train_x - bounds[0]) / (bounds[1] - bounds[0])

    # Standardize the outputs
    train_y_standardized = standardize(train_y)

    # Initialize models for each output
    model1 = SingleTaskGP(train_x_scaled, train_y_standardized[:, 0:1])
    model2 = SingleTaskGP(train_x_scaled, train_y_standardized[:, 1:2])

    # Optimize the model hyperparameters
    mll1 = ExactMarginalLogLikelihood(model1.likelihood, model1)
    mll2 = ExactMarginalLogLikelihood(model2.likelihood, model2)

    fit_gpytorch_model(mll1)
    fit_gpytorch_model(mll2)

    models = [model1, model2]

    return models, train_y.mean(dim=0), train_y.std(dim=0)


def get_ucb_candidate(models, train_x, train_y, y_mean, y_std, n_pareto_samples=8):
    """Determine what experiment to do next using GP-UCB."""
    train_x_tensor = torch.tensor(train_x, dtype=torch.float64)
    bounds = torch.tensor(BOUNDS, dtype=torch.float64)

    best_candidates = []

    # Generate a set of candidate points with different strategies
    for i in range(n_pareto_samples):
        beta = 1.0 + np.random.random() * 2.0  # Between 1.0 and 3.0, randomized

        # Alternate between models to balance exploration of both objectives
        model_idx = i % 2  
        ucb = [UpperConfidenceBound(models[0], beta=beta), UpperConfidenceBound(models[1], beta=beta)]

        # Use unit cube bounds for BoTorch
        unit_bounds = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],  # Lower bounds (scaled)
            [1.0, 1.0, 1.0, 1.0]  # Upper bounds (scaled)
        ], dtype=torch.float64)

        candidate, acq0 = optimize_acqf(
            acq_function=ucb[model_idx],
            bounds=unit_bounds,
            q=1,
            num_restarts=20,  
            raw_samples=2048,  
        )

        other_model_idx = (model_idx + 1) % 2
        # Calculate acquisition value for the other model
        _, acq1 = optimize_acqf(
            acq_function=ucb[other_model_idx],
            bounds=unit_bounds,
            q=1,
            num_restarts=20,  
            raw_samples=2048,  
        )

        # Destandardize acquisition values 
        acq_values = [acq0.detach().item(), acq1.detach().item()]
        acq_values = np.array(acq_values)
        acq_values_destandardized = acq_values * y_std.numpy() + y_mean.numpy()
        acq_values_destandardized[1] = 1.0 / acq_values_destandardized[1]  # Inverse for Energy

        # Scale back to original bounds
        orig_bounds = torch.tensor(BOUNDS, dtype=torch.float64)
        candidate = candidate * (orig_bounds[1] - orig_bounds[0]) + orig_bounds[0]

        best_candidates.append([candidate.detach(), acq_values_destandardized, model_idx])

    # Choose a diverse set by selecting the most different point from already explored points
    best_candidate = best_candidates[0]
    max_min_distance = -float('inf')

    for candidate in best_candidates:
        min_distance = float('inf')
        for x in train_x_tensor:
            # Calculate normalized distance 
            normalized_diff = (candidate[0] - x) / (bounds[1] - bounds[0])
            dist = torch.norm(normalized_diff)
            min_distance = min(min_distance, dist.item())

        if min_distance > max_min_distance:
            max_min_distance = min_distance
            best_candidate = candidate

    # Add a small random perturbation to avoid identical experiments if too close to an existing point
    if max_min_distance < 0.15:  
        perturbation = torch.zeros_like(best_candidate[0])
        for i in range(4):
            scale = (bounds[1, i] - bounds[0, i]) * (0.02 + 0.03 * np.random.random())
            perturbation[0, i] = (np.random.random() * 2 - 1) * scale

        best_candidate[0] = best_candidate[0] + perturbation
        best_candidate[0] = torch.max(torch.min(best_candidate[0], bounds[1]), bounds[0])

    # Round temperature and time values to integers
    best_candidate_np = best_candidate[0].squeeze().numpy()
    best_candidate_np[0] = np.round(best_candidate_np[0])  # T1
    best_candidate_np[1] = np.round(best_candidate_np[1])  # t1
    best_candidate_np[2] = np.round(best_candidate_np[2])  # T2
    best_candidate_np[3] = np.round(best_candidate_np[3])  # t2

    return best_candidate_np


# Termination Criteria - user input
def check_termination():
    """Check if user wants to terminate."""
    user_input = input("\nEnter 'stop' to end optimization or press Enter to continue: ")
    if user_input.lower() == 'stop':
        return True
    return False



def get_user_experiment(next_experiment):
    """Get experimental results from user."""
    print(f"\nEXPERIMENTAL RESULTS INPUT")
    print("Please run the experiment with the above parameters and enter the results:")
    
    while True:
        try:
            uts_measured = float(input(f"Enter the measured UTS (MPa): "))
            if uts_measured <= 0:
                print(" UTS must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value for UTS.")
    
    energy_measured, energy_per_hour = energy_consumption(
        next_experiment[0], next_experiment[2], next_experiment[1], next_experiment[3]
    )
    
    print(f" Calculated energy consumption: {energy_measured:.3f} kWh")
    print(f" Energy per hour: {energy_per_hour:.3f} kW")
    
    return uts_measured, energy_measured, energy_per_hour


def identify_pareto_front(train_x, train_y_transformed, train_y_original):
    """Identify the Pareto front from the observed points."""
    train_y_tensor = torch.tensor(train_y_transformed, dtype=torch.float64)
    pareto_mask = is_non_dominated(train_y_tensor)
    pareto_y = train_y_transformed[pareto_mask.numpy()]
    pareto_y_original = train_y_original[pareto_mask.numpy()]
    pareto_x = train_x[pareto_mask.numpy()]

    return pareto_x, pareto_y, pareto_y_original

def plot_pareto_front(train_x, train_y_transformed, train_y_original, pareto_x, pareto_y, pareto_y_original, iteration):
    """Plot the Pareto front with parameter details."""
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()

    ax1.scatter(train_y_transformed[:, 0], train_y_transformed[:, 1], c='lightgray',
                label='All experiments', alpha=0.7, s=100)
        
    ax1.scatter(pareto_y[:, 0], pareto_y[:, 1], facecolors='white', edgecolors='orange', 
            linewidths=1.2, label='Pareto optimal', s=150, zorder=10)
    
    n_initial = len(config['initial_process_parameters']['T1 (C)'])
    ax1.scatter(train_y_transformed[:n_initial, 0], train_y_transformed[:n_initial, 1], color='blue',
                label='Initial points', s=120, zorder=9)

    for i, (x, y) in enumerate(pareto_y_original):
        ax1.annotate(f'P{i + 1}', (x, y), textcoords="offset points", xytext=(5, 5),
                     ha='center', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))

    for i in range(min(n_initial, len(train_y_original))):
        ax1.annotate(f'I{i + 1}', (train_y_original[i, 0], train_y_original[i, 1]),
                     textcoords="offset points", xytext=(5, 5),
                     ha='center', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
    

    if len(pareto_y) > 1:
        sorted_indices = np.argsort(pareto_y[:, 0])
        x_step = pareto_y[sorted_indices, 0]
        y_step = pareto_y[sorted_indices, 1]
        x_left_edge = np.min(train_y_transformed[:, 0]) * 0.98
        y_base = 0.0
        x_fill = np.r_[x_left_edge, x_step]
        y_fill = np.r_[y_step[0], y_step]
        ax1.fill_between(x_fill, y_fill, y_base, step='pre',
                        color='orange', alpha=0.15, linewidth=0, zorder=1)
        xs, ys = [], []
        for i in range(len(x_step) - 1):
            xs.extend([x_step[i], x_step[i], x_step[i + 1]])
            ys.extend([y_step[i], y_step[i + 1], y_step[i + 1]])
        ax1.plot(xs, ys, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        
        idx_ymax = np.argmax(y_step)
        x_at_ymax = x_step[idx_ymax]
        y_max = y_step[idx_ymax]
        x_max = x_step[np.argmax(x_step)]
        
        ax1.plot([x_left_edge, x_at_ymax], [y_max, y_max],
                linestyle='--', color='orange', linewidth=1.5, alpha=0.7)
        
        ax1.plot([x_max, x_max], [0.0, y_step[np.argmax(x_step)]],
                linestyle='--', color='orange', linewidth=1.5, alpha=0.7)


    ax1.set_xlabel('UTS (MPa)', fontsize=14)
    ax1.set_ylabel('1/E (1/kWh)', fontsize=14)
    ax1.legend(fontsize=12)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    pareto_indices = []
    for p_point in pareto_x:
        for i, point in enumerate(train_x):
            if np.allclose(p_point, point, rtol=1e-5, atol=1e-5):
                pareto_indices.append(i)
                break
    
    pareto_data = {
        "ID": [],
        "T1 (C)": [],
        "t1 (min)": [],
        "T2 (C)": [],
        "t2 (min)": [],
        "Ultimate Tensile Strength (MPa)": [],
        "Energy (kWh)": []
    }

    for i, idx in enumerate(pareto_indices):
        pareto_data["ID"].append(f'P{i + 1}')
        pareto_data["T1 (C)"].append(f'{train_x[idx, 0]:.0f}')
        pareto_data["t1 (min)"].append(f'{train_x[idx, 1]:.1f}')
        pareto_data["T2 (C)"].append(f'{train_x[idx, 2]:.0f}')
        pareto_data["t2 (min)"].append(f'{train_x[idx, 3]:.1f}')
        pareto_data["Ultimate Tensile Strength (MPa)"].append(f'{train_y_original[idx, 0]:.1f}')
        pareto_data["Energy (kWh)"].append(f'{train_y_original[idx, 1]:.2f}')

    new_pareto_df = pl.DataFrame(pareto_data)
    new_pareto_df.write_csv(CSV_PATH)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, f'pareto_front_iter_{iteration}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("\n=== PARETO-OPTIMAL EXPERIMENT PARAMETERS ===")
    print("ID  T1(C)  t1(min)  T2(C)  t2(min)  UTS(MPa)  Energy(kWh)")
    print("-" * 60)

    for i, idx in enumerate(pareto_indices):
        print(
            f"P{i + 1}  {train_x[idx, 0]:.0f}   {train_x[idx, 1]:.1f}    {train_x[idx, 2]:.0f}   {train_x[idx, 3]:.1f}    {train_y_original[idx, 0]:.1f}      {train_y_original[idx, 1]:.2f}")

def plot_hypervolume_progress(iterations, hypervolumes, iteration_num=None):
    """Plot the progress of hypervolume improvement over iterations."""
    plt.figure(figsize=(10, 6))
    plt.step(iterations, hypervolumes, where='post', label='Hypervolume', linewidth=2, marker='o', markersize=6)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.title('Optimization Progress', fontsize=14)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.text(
        0.975, 0.04, f"{hypervolumes[-1]:.2f}",
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="black", alpha=0.9, linewidth=0.8)
    )

    if iteration_num is not None:
        filename = f'hypervolume_progress_iter_{iteration_num}.png'
    else:
        filename = 'hypervolume_progress_final.png'

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, filename), dpi=300, bbox_inches='tight')
    plt.show()

def calculate_hypervolume(pareto_y):
    """Calculate hypervolume indicator for the Pareto front."""

    if len(pareto_y) <= 0:
        return 0.0

    reference_point = REFERENCE

    sorted_indices = np.argsort(-pareto_y[:, 0])
    sorted_pareto = pareto_y[sorted_indices]

    hypervolume = 0.0
    max_inv_energy_so_far = reference_point[1]

    for point in sorted_pareto:
        # consider points that contribute to hypervolume
        if point[1] > max_inv_energy_so_far:
            width = point[0] - reference_point[0]
            height = point[1] - max_inv_energy_so_far

            hypervolume += width * height

            max_inv_energy_so_far = point[1]

    return hypervolume


def initialize_optimization():
    """Load data and calculate starting state for optimization."""
    # Load previous experiments first if any
    train_x, train_y_transformed, train_y_original = load_experiments(DATABASE_PATH)
    
    if train_x is None:
        train_x, train_y_transformed, train_y_original = load_data()
        initialize_database(DATABASE_PATH, (train_x, train_y_transformed, train_y_original))
    
    # Track all experiments and results
    all_experiments = train_x.copy()
    all_results_transformed = train_y_transformed.copy()
    all_results_original = train_y_original.copy()
    
    # Calculate starting iteration number
    n_initial = len(config['initial_process_parameters']['T1 (C)'])
    starting_iteration = max(0, len(all_experiments) - n_initial)
    
    # Calculate hypervolumes for all existing experiments
    hypervolumes = []
    for i in range(n_initial, len(all_experiments) + 1):
        subset_x = all_experiments[:i]
        subset_y_transformed = all_results_transformed[:i]
        subset_y_original = all_results_original[:i]
        pareto_x, pareto_y, _ = identify_pareto_front(subset_x, subset_y_transformed, subset_y_original)
        hv = calculate_hypervolume(pareto_y)
        hypervolumes.append(hv)
    
    # Print starting message
    if starting_iteration > 0:
        print(f" Resuming from iteration {starting_iteration}")
        print(f"   Total experiments so far: {len(all_experiments)}")
        print(f"   Current hypervolume: {hypervolumes[-1]:.4f}\n")
    else:
        print(f" Starting fresh optimization")
        print(f"   Initial experiments: {n_initial}")
        print(f"   Initial hypervolume: {hypervolumes[-1]:.4f}\n")
    
    return all_experiments, all_results_transformed, all_results_original, starting_iteration, hypervolumes, n_initial


def save_experiment_to_database(next_experiment, uts_measured, energy_measured):
    """Save new experiment to database CSV."""
    
    new_row = pl.DataFrame({
        "T1 (C)": [next_experiment[0]],
        "t1 (min)": [next_experiment[1]], 
        "T2 (C)": [next_experiment[2]],
        "t2 (min)": [next_experiment[3]],
        "Ultimate Tensile Strength (MPa)": [uts_measured],
        "Energy (kWh)": [energy_measured]
    })
    
    existing_df = pl.read_csv(DATABASE_PATH, schema_overrides={
        "T1 (C)": pl.Float64,
        "t1 (min)": pl.Float64,
        "T2 (C)": pl.Float64,
        "t2 (min)": pl.Float64,
        "Ultimate Tensile Strength (MPa)": pl.Float64,
        "Energy (kWh)": pl.Float64
    })
    updated_df = pl.concat([existing_df, new_row])
    updated_df.write_csv(DATABASE_PATH)


def update_experiment_arrays(all_experiments, all_results_original, all_results_transformed, 
                             next_experiment, uts_measured, energy_measured):
    """Add new experiment to tracking arrays."""
    new_experiment = next_experiment.reshape(1, -1)
    new_result_original = np.array([[uts_measured, energy_measured]])
    new_result_transformed = np.array([[uts_measured, 1.0 / energy_measured]])
    
    all_experiments = np.vstack([all_experiments, new_experiment])
    all_results_original = np.vstack([all_results_original, new_result_original])
    all_results_transformed = np.vstack([all_results_transformed, new_result_transformed])
    
    return all_experiments, all_results_original, all_results_transformed


def display_current_status(all_experiments, pareto_x, current_hypervolume):
    """Print current optimization status and Pareto points."""
    print(f"\n=== CURRENT STATUS ===")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Pareto-optimal points: {len(pareto_x)}")
    print(f"Current hypervolume: {current_hypervolume:.4f}")
   

def display_completion_summary(iteration, all_experiments, current_hypervolume, pareto_x):
    """Print optimization completion summary."""
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total iterations: {iteration}")
    print(f"Total experiments: {len(all_experiments)}")
    print(f"Final hypervolume: {current_hypervolume:.4f}")
    print(f"Final Pareto front has {len(pareto_x)} points")
    print(f"   You can resume this optimization later by running the script again")


def run_mobo_loop():
    """Run the full interactive Bayesian optimization loop."""
    # Initialize optimization
    all_experiments, all_results_transformed, all_results_original, iteration, hypervolumes, n_initial = initialize_optimization()
    
    # Start optimization loop
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"OPTIMIZATION ITERATION {iteration}")
        print(f"{'='*60}")
        
        # Train models and get next experiment suggestion (GP-UCB)
        models, y_mean, y_std = initialize_GP_model(all_experiments, all_results_transformed)
        next_experiment = get_ucb_candidate(models, all_experiments, all_results_transformed, y_mean, y_std)
        
        print(f"RECOMMENDED NEXT EXPERIMENT")
        print(f"{'='*60}")
        print(f"First heat treatment:  T1 = {int(next_experiment[0])}°C, t1 = {next_experiment[1]:.1f} minutes")
        print(f"Second heat treatment: T2 = {int(next_experiment[2])}°C, t2 = {next_experiment[3]:.1f} minutes")
        print(f"{'='*60}")
        
        # Get user experimental results
        uts_measured, energy_measured, energy_per_hour = get_user_experiment(next_experiment)
        
        # Update experiment arrays
        all_experiments, all_results_original, all_results_transformed = update_experiment_arrays(
            all_experiments, all_results_original, all_results_transformed, 
            next_experiment, uts_measured, energy_measured
        )
        
        # Save to database
        save_experiment_to_database(next_experiment, uts_measured, energy_measured)
        
        # Analyze current Pareto front
        pareto_x, pareto_y, pareto_y_original = identify_pareto_front(
            all_experiments, all_results_transformed, all_results_original
        )
        
        # Calculate and store hypervolume
        current_hypervolume = calculate_hypervolume(pareto_y)
        hypervolumes.append(current_hypervolume)
        
        # Display current status
        display_current_status(all_experiments, pareto_x, current_hypervolume)
        
        # Generate plots
        plot_pareto_front(all_experiments, all_results_transformed, all_results_original, 
                         pareto_x, pareto_y, pareto_y_original, iteration)
        plot_hypervolume_progress(np.arange(1, len(hypervolumes) + 1), hypervolumes, iteration)

        print(f"\n{'='*60}")
        
        if check_termination():
            display_completion_summary(iteration, all_experiments, current_hypervolume, pareto_x)
            break
    
    return all_experiments, all_results_transformed, all_results_original, pareto_x, pareto_y_original

if __name__ == "__main__":
    run_mobo_loop()
