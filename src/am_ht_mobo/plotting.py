import numpy as np
import polars as pl
import torch
import os
import seaborn as sns
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, r2_score
from . import bo
from .filemanager import get_file_manager
import matplotlib.patches as mpatches

# Get paths from filemanager
fm = get_file_manager()
config = fm.config
paths = fm.get_paths()
METRIC_FIGURES_PATH = paths['figures']
NUM_INITIALS = len(config['initial_process_parameters']['T1 (C)'])  
PARAM_NAMES = ['T$_{SA}$ (°C)', 't$_{SA}$ (min)', 'T$_{AA}$ (°C)', 't$_{AA}$ (min)']

def load_data_from_database():
    """Load experiment data from csv file."""
    # Load data from database
    df = pl.read_csv(bo.DATABASE_PATH)
    
    if df.height == 0:
        raise ValueError("Database is empty. Run some optimization experiments first.")
    
    # Get experiment data from database
    points = df.select(["T1 (C)", "t1 (min)", "T2 (C)", "t2 (min)"]).to_numpy()
    results_original = df.select(["Ultimate Tensile Strength (MPa)", "Energy (kWh)"]).to_numpy()
    
    results_transformed = np.copy(results_original)
    results_transformed[:, 1] = 1.0 / results_original[:, 1]
    
    return points, results_transformed, results_original

def loocv_get_results(train_x, train_y_transformed, train_y_original, bounds):
    """Calculate LOOCV predictions and uncertainties for GP model validation."""
    n_experiments = len(train_x)

    train_x_tensor = torch.tensor(train_x, dtype=torch.float64)
    train_y_tensor = torch.tensor(train_y_transformed, dtype=torch.float64)
    bounds_tensor = torch.tensor(bounds, dtype=torch.float64)
    train_x_scaled = (train_x_tensor - bounds_tensor[0]) / (bounds_tensor[1] - bounds_tensor[0])
    train_y_standardized = standardize(train_y_tensor)

    results = {
        'UTS': {'predictions': [], 'actuals': [], 'uncertainties': [], 'parameters': []},
        'energy': {'predictions': [], 'actuals': [], 'uncertainties': [], 'parameters': []}
    }


    for i in range(n_experiments):

        train_indices = [j for j in range(n_experiments) if j != i]
        loocv_train_x = train_x_scaled[train_indices]
        loocv_train_y = train_y_standardized[train_indices]
        loocv_test_x = train_x_scaled[i:i + 1]

        for obj_idx, obj_name in enumerate(['UTS', 'energy']):
            model = SingleTaskGP(loocv_train_x, loocv_train_y[:, obj_idx:obj_idx + 1])
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)

            model.eval()
            with torch.no_grad():
                posterior = model(loocv_test_x)
                pred_mean_std = posterior.mean.item()
                pred_var_std = posterior.variance.item()
                pred_std_std = np.sqrt(pred_var_std)

            if obj_idx == 0:
                pred_original = pred_mean_std * train_y_tensor[:, 0].std() + train_y_tensor[:, 0].mean()
                uncertainty_original = pred_std_std * train_y_tensor[:, 0].std()
                actual_original = train_y_original[i, 0]
            else:
                pred_inv_energy = pred_mean_std * train_y_tensor[:, 1].std() + train_y_tensor[:, 1].mean()
                pred_original = 1.0 / pred_inv_energy if pred_inv_energy > 0 else np.inf
                uncertainty_original = pred_std_std * train_y_tensor[:, 1].std() / (pred_inv_energy ** 2)
                actual_original = train_y_original[i, 1]

            results[obj_name]['predictions'].append(pred_original)
            results[obj_name]['actuals'].append(actual_original)
            results[obj_name]['uncertainties'].append(uncertainty_original)
            results[obj_name]['parameters'].append(train_x[i].copy())

    return results

def loocv_validation_detailed(train_x, train_y_transformed, train_y_original, bounds):
    """Generate LOOCV parity plots to validate GP model accuracy."""
    print("=" * 80)
    print("LOOCV PARITY PLOTS - MODEL VALIDATION")
    print("=" * 80)
    print("Purpose: Validate GP surrogate models used in Bayesian optimization")
    print("Method: Leave-one-out cross-validation")
    print("Shows: Prediction accuracy and uncertainty calibration")
    print()

    results = loocv_get_results(train_x, train_y_transformed, train_y_original, bounds)

    for obj_idx, obj_name in enumerate(['UTS', 'energy']):
        print(f"\nGenerating {obj_name.capitalize()} LOOCV Plot...")

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        predictions = np.array(results[obj_name]['predictions'])
        actuals = np.array(results[obj_name]['actuals'])
        uncertainties = np.array(results[obj_name]['uncertainties'])

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        relative_rmse = (rmse / np.mean(actuals)) * 100
        r2 = r2_score(actuals, predictions)

        ax.errorbar(actuals, predictions, yerr=2 * uncertainties, fmt='o', capsize=3, capthick=1, markersize=6,
                    alpha=0.7, label=f'LOOCV predictions ±2σ')

        min_val = min(np.min(actuals), np.min(predictions))
        max_val = max(np.max(actuals), np.max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if obj_name == 'UTS':
            ax.set_xlabel('Experimental UTS (MPa)', fontsize=10)
            ax.set_ylabel('Predicted UTS (MPa)', fontsize=10)
        else:
            ax.set_xlabel(f'Experimental {obj_name.capitalize()} (kWh)', fontsize=10)
            ax.set_ylabel(f'Predicted {obj_name.capitalize()} (kWh)', fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.legend(fontsize=8)

        filename = f'{METRIC_FIGURES_PATH}/loocv_{obj_name}_validation.png'
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Saved: {filename}")

    print("\nINTERPRETATION:")
    print("- Points close to red line = accurate predictions")
    print("- Small error bars = confident predictions")
    print("- Large error bars = uncertain predictions")
    print("- This validates that GP models can predict properties from heat treatment parameters")

    print("\nGenerated files:")
    print("- loocv_UTS_validation.png")
    print("- loocv_energy_validation.png")

def uncertainty_heatmaps(train_x, train_y_transformed, train_y_original, bounds, pareto_colors):
    """Generate uncertainty heatmaps showing GP model confidence across parameter space."""
    print("=" * 80)
    print("UNCERTAINTY HEATMAPS")
    print("=" * 80)
    print("Purpose: Show where GP models are confident vs uncertain for each property")
    print("Method: 2D slices through 4D parameter space")
    print("Shows: Dark purple = confident, Bright yellow = uncertain")
    print()

    models, _, _ = bo.initialize_GP_model(train_x, train_y_transformed)
    pareto_x, pareto_y, pareto_y_original = bo.identify_pareto_front(
        train_x, train_y_transformed, train_y_original
    )

    pareto_df = get_pareto_df()
    pareto_names = ["P" + str(s) for s in pareto_df.select("Sample number").to_numpy().flatten()]
    pareto_colors_plot = _prepare_pareto_colors(pareto_colors, len(pareto_x))

    slice_configs = [
        {
            'vary': [0, 1], 
            'fix': {2: 750, 3: 300}, 
            'cbar_lim': {'vmin': 0, 'vmax': 0.5}, 
            'title': 'Solution Heat Treatment'
        },
        {
            'vary': [2, 3], 
            'fix': {0: 1100, 1: 60}, 
            'cbar_lim': {'vmin': 0, 'vmax': 0.5}, 
            'title': 'Aging Treatment'
        }
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    for obj_idx, (model, obj_name) in enumerate(zip(models, ['UTS', 'Energy'])):

        for slice_idx, config in enumerate(slice_configs):
            ax = axes[obj_idx][slice_idx]
            vary_params = config['vary']
            fix_params = config['fix']

            
            n_points = 50
            param1_margin = (bounds[1][vary_params[0]] - bounds[0][vary_params[0]]) * 0.02
            param2_margin = (bounds[1][vary_params[1]] - bounds[0][vary_params[1]]) * 0.02
            
            param1_range = np.linspace(
                bounds[0][vary_params[0]] - param1_margin,
                bounds[1][vary_params[0]] + param1_margin, 
                n_points
            )
            param2_range = np.linspace(
                bounds[0][vary_params[1]] - param2_margin,
                bounds[1][vary_params[1]] + param2_margin, 
                n_points
            )

            # Generate test points
            test_points = []
            for p1 in param1_range:
                for p2 in param2_range:
                    point = [0, 0, 0, 0]
                    point[vary_params[0]] = p1
                    point[vary_params[1]] = p2
                    for fixed_idx, fixed_val in fix_params.items():
                        point[fixed_idx] = fixed_val
                    test_points.append(point)

            # Scale test points to unit cube
            test_x = np.array(test_points)
            test_x_tensor = torch.tensor(test_x, dtype=torch.float64)
            bounds_tensor = torch.tensor(bounds, dtype=torch.float64)
            test_x_scaled = (test_x_tensor - bounds_tensor[0]) / (bounds_tensor[1] - bounds_tensor[0])

            # Calculate uncertainties
            uncertainties = []
            model.eval()
            with torch.no_grad():
                for point_scaled in test_x_scaled:
                    posterior = model(point_scaled.unsqueeze(0))
                    uncertainty = torch.sqrt(posterior.variance).item()
                    uncertainties.append(uncertainty)

            uncertainty_grid = np.array(uncertainties).reshape(n_points, n_points)

            # Plot uncertainty heatmap
            levels = np.linspace(config["cbar_lim"]["vmin"], config["cbar_lim"]["vmax"], 25)
            im = ax.contourf(
                param1_range, param2_range, uncertainty_grid.T, 
                levels=levels, cmap='plasma', alpha=0.8, extend='both'
            )

            # Plot all experiments
            ax.scatter(
                train_x[:, vary_params[0]], train_x[:, vary_params[1]], 
                c='white', s=80, marker='o', linewidth=2, alpha=0.7, 
                edgecolor='black', zorder=10
            )

            # Plot Pareto-optimal points
            if len(pareto_x) > 0:
                ax.scatter(
                    pareto_x[:, vary_params[0]], pareto_x[:, vary_params[1]], 
                    c=pareto_colors_plot, s=120, marker='X', linewidth=2, 
                    edgecolor='black', zorder=11
                )

            # Set labels and title
            ax.set_xlabel(PARAM_NAMES[vary_params[0]], fontsize=12)
            ax.set_ylabel(PARAM_NAMES[vary_params[1]], fontsize=12)
            ax.set_title(f'{obj_name}', fontsize=12)
            ax.set_xlim(bounds[0][vary_params[0]] - param1_margin, bounds[1][vary_params[0]] + param1_margin)
            ax.set_ylim(bounds[0][vary_params[1]] - param2_margin, bounds[1][vary_params[1]] + param2_margin)

            if obj_idx == 0 and slice_idx == 0:
                _add_uncertainty_legend(ax, pareto_colors_plot, pareto_names)

            im.set_clim(config["cbar_lim"]["vmin"], config["cbar_lim"]["vmax"])

            mean_uncertainty = np.mean(uncertainties)
            std_uncertainty = np.std(uncertainties)
            stats_text = f'Mean σ: {mean_uncertainty:.3f}\nStd σ: {std_uncertainty:.3f}'
            ax.text(
                0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

    plt.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.03])
    vmin = min([cfg["cbar_lim"]["vmin"] for cfg in slice_configs])
    vmax = max([cfg["cbar_lim"]["vmax"] for cfg in slice_configs])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', extend='both')
    cbar.set_label('Prediction Uncertainty (σ)', fontsize=12)
    cbar.set_ticks(np.linspace(vmin, vmax, num=11))
    cbar.set_ticklabels([f"{tick:.2f}" for tick in np.linspace(vmin, vmax, num=11)])

    filename = f'{METRIC_FIGURES_PATH}/uncertainty_heatmaps.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print("\nINTERPRETATION:")
    print("- Dark purple regions = model is confident (low uncertainty)")
    print("- Bright yellow regions = model is uncertain (needs more experiments)")
    print("\nGenerated files:")
    print("- uncertainty_heatmaps.png")


def _prepare_pareto_colors(pareto_colors, n_pareto):
    """Helper function to ensure pareto_colors matches number of Pareto points."""
    if len(pareto_colors) == n_pareto:
        return pareto_colors
    
    if len(pareto_colors) > n_pareto:
        return pareto_colors[:n_pareto]
    
    repeats = (n_pareto + len(pareto_colors) - 1) // len(pareto_colors)
    return (pareto_colors * repeats)[:n_pareto]


def _add_uncertainty_legend(ax, pareto_colors_plot, pareto_names):
    """Helper function to add legend with all experiments and Pareto points."""
    
    num_items = min(len(pareto_names), len(pareto_colors_plot))
    
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markeredgecolor='black', markersize=8, linewidth=0, 
               label='All Points', markeredgewidth=2)
    ]
    handles.extend([
        mpatches.Patch(color=pareto_colors_plot[i], label=pareto_names[i]) 
        for i in range(num_items)
    ])
    
    ax.legend(handles=handles, fontsize=10, loc='upper right')

def plot_parameter_objective_correlations(train_x, train_y_original, pareto_x, pareto_y_original, pareto_colors):
    """Plot correlation between process parameters and objectives (UTS and energy)."""
    print("=" * 70)
    print("PARAMETER-OBJECTIVE CORRELATION ANALYSIS")
    print("=" * 70)
    print("Purpose: Show how each parameter affects UTS and energy consumption")
    print("Shows: Individual parameter sensitivities and optimal parameter ranges")
    print()

    objective_names = ['UTS (MPa)', 'Energy Consumption (kWh)']

    pareto_df = get_pareto_df()
    pareto_names = ["P" + str(s) for s in pareto_df.select("Sample number").to_numpy().flatten()]
    pareto_colors_plot = _prepare_pareto_colors(pareto_colors, len(pareto_x))

    for obj_idx in range(2):

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))

        for param_idx in range(4):
            row = param_idx // 2
            col = param_idx % 2
            ax = axes[row, col]

            param_data = train_x[:, param_idx]
            obj_data = train_y_original[:, obj_idx]

            ax.scatter(param_data, obj_data, color='blue', alpha=0.6, s=80, edgecolor='black', linewidth=0.5,
                       label='All experiments')

            if len(pareto_x) > 0:
                pareto_param = pareto_x[:, param_idx]
                pareto_obj = pareto_y_original[:, obj_idx]

                ax.scatter(pareto_param, pareto_obj, color=pareto_colors_plot, s=120, marker='X', edgecolor=pareto_colors_plot,
                           label='Pareto optimal', zorder=10)

                if row == 0 and col == 0:
                    handles = [mpatches.Patch(color=pareto_colors_plot[i], label=pareto_names[i]) for i in range(len(pareto_names))]
                    ax.legend(handles=handles, fontsize=9, loc='upper right', title='Pareto Points')

            correlation = np.corrcoef(param_data, obj_data)[0, 1]

            title = f'ρ = {correlation:.3f}'

            ax.set_xlabel(PARAM_NAMES[param_idx], fontsize=11)
            ax.set_ylabel(objective_names[obj_idx], fontsize=11)
            ax.set_title(title, fontsize=10)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        objective_name_clean = objective_names[obj_idx].replace(' ', '_').replace('(', '').replace(')', '').replace('/',
                                                                                                                    '_')
        plt.suptitle(f'Parameter Effects on {objective_names[obj_idx]}\n', fontsize=14)
        plt.tight_layout()

        filename = f'{METRIC_FIGURES_PATH}/parameter_{objective_name_clean.lower()}_correlations.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Generated: {filename}")

    print("\nDETAILED CORRELATION ANALYSIS:")
    print("=" * 50)

    UTS_correlations = []
    energy_correlations = []

    for param_idx in range(4):
        UTS_corr = np.corrcoef(train_x[:, param_idx], train_y_original[:, 0])[0, 1]
        energy_corr = np.corrcoef(train_x[:, param_idx], train_y_original[:, 1])[0, 1]

        UTS_correlations.append((PARAM_NAMES[param_idx], UTS_corr))
        energy_correlations.append((PARAM_NAMES[param_idx], energy_corr))

    print("\nUTS SENSITIVITY:")
    print("-" * 25)
    UTS_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for param, corr in UTS_correlations:
        direction = "increases" if corr > 0 else "decreases"
        strength = "strongly" if abs(corr) > 0.5 else "moderately" if abs(corr) > 0.3 else "weakly"
        print(f"{param:12}: ρ = {corr:+.3f} → UTS {strength} {direction}")

    print("\nENERGY CONSUMPTION SENSITIVITY:")
    print("-" * 30)
    energy_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    for param, corr in energy_correlations:
        direction = "increases" if corr > 0 else "decreases"
        strength = "strongly" if abs(corr) > 0.5 else "moderately" if abs(corr) > 0.3 else "weakly"
        print(f"{param:12}: ρ = {corr:+.3f} → energy {strength} {direction}")

    print("\nKEY FINDINGS:")
    print("-" * 15)

    most_influential_UTS = max(UTS_correlations, key=lambda x: abs(x[1]))
    print(f"• Most influential for UTS: {most_influential_UTS[0]} (ρ = {most_influential_UTS[1]:+.3f})")

    most_influential_energy = max(energy_correlations, key=lambda x: abs(x[1]))
    print(f"• Most influential for ENERGY: {most_influential_energy[0]} (ρ = {most_influential_energy[1]:+.3f})")

    print(f"\nCONFLICTING PARAMETERS (opposite effects on objectives):")
    print("-" * 50)
    for i, param in enumerate(PARAM_NAMES):
        h_corr = UTS_correlations[i][1] if UTS_correlations[i][0] == param else next(
            x[1] for x in UTS_correlations if x[0] == param)
        e_corr = energy_correlations[i][1] if energy_correlations[i][0] == param else next(
            x[1] for x in energy_correlations if x[0] == param)

        if (h_corr * e_corr < 0) and (abs(h_corr) > 0.2 or abs(e_corr) > 0.2):
            h_effect = "increases" if h_corr > 0 else "decreases"
            e_effect = "increases" if e_corr > 0 else "decreases"
            print(f"• {param}: {h_effect} UTS ({h_corr:+.3f}) but {e_effect} energy ({e_corr:+.3f})")

    print(f"\nGenerated files:")
    print("- parameter_UTS_correlations.png")
    print("- parameter_energy_consumption_kwh_correlations.png")

def plot_parameter_importance_summary(train_x, train_y_original):
    """Generate heatmap showing parameter importance for each objective."""
    print(f"\nGenerating parameter importance heatmap...")

    objective_names = ['UTS', 'Energy']

    correlation_matrix = np.zeros((4, 2))

    for param_idx in range(4):
        for obj_idx in range(2):
            correlation_matrix[param_idx, obj_idx] = np.corrcoef(train_x[:, param_idx], train_y_original[:, obj_idx])[
                0, 1]

    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='plasma', vmin=-1, vmax=1, 
                     annot_kws={"size": 14}, linewidths=0.75,
                     cbar_kws={'label': 'Correlation Coefficient (ρ)'})
    
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(14)
    cbar.ax.yaxis.set_label_coords(4, 0.5)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xticklabels(objective_names, fontsize=14)
    ax.set_yticklabels(PARAM_NAMES, fontsize=14, rotation=0)

    plt.tight_layout()
    plt.savefig(f'{METRIC_FIGURES_PATH}/parameter_importance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Generated: {METRIC_FIGURES_PATH}/parameter_importance_heatmap.png")

def get_pareto_df():
    """Load data from database and return DataFrame with Pareto front data."""
    train_x, train_y_transformed, train_y_original = load_data_from_database()
    
    pareto_x, pareto_y, pareto_y_original = bo.identify_pareto_front(
        train_x, train_y_transformed, train_y_original
    )
    
    pareto_indices = []
    for px in pareto_x:
        matches = np.all(np.isclose(train_x, px, rtol=1e-5), axis=1)  
        idx = np.where(matches)[0]
        if len(idx) > 0:
            pareto_indices.append(idx[0])
    
    if pareto_indices:
        pareto_data = {
            "T1 (C)": train_x[pareto_indices, 0],
            "t1 (min)": train_x[pareto_indices, 1],
            "T2 (C)": train_x[pareto_indices, 2],
            "t2 (min)": train_x[pareto_indices, 3],
            "Ultimate Tensile Strength (MPa)": train_y_original[pareto_indices, 0],
            "Energy (kWh)": train_y_original[pareto_indices, 1],
            "Sample number": list(range(1, len(pareto_indices) + 1))
        }
        pareto_df = pl.DataFrame(pareto_data)
    else:
        pareto_df = pl.DataFrame({
            "T1 (C)": [],
            "t1 (min)": [],
            "T2 (C)": [],
            "t2 (min)": [],
            "Ultimate Tensile Strength (MPa)": [],
            "Energy (kWh)": [],
            "Sample number": []
        })
    
    return pareto_df

def plot_experiment_distribution(train_x, train_y_transformed, train_y_original, c_blind_5254):
    """Plot the distribution of experiments in parameter space with Pareto front highlighted."""
    # Identify Pareto front
    pareto_x, pareto_y, pareto_y_original = bo.identify_pareto_front(train_x, train_y_transformed, train_y_original)
    
    # Get Pareto colors
    pareto_colors_plot = _prepare_pareto_colors(c_blind_5254, len(pareto_x))
    pareto_df = get_pareto_df()
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # First Heat Treatment (T1 vs t1)
    ax = axs[0, 0]
    ax.scatter(train_x[:, 0], train_x[:, 1], color='blue', alpha=0.6, s=80, 
               edgecolor='black', linewidth=0.5, label='All experiments')
    ax.scatter(pareto_x[:, 0], pareto_x[:, 1], color=pareto_colors_plot, s=120, 
               marker='X', edgecolor=pareto_colors_plot, label='Pareto optimal', zorder=10)
    
    pareto_names = ["P" + str(s) for s in pareto_df.select("Sample number").to_numpy().flatten()]
    handles = [mpatches.Patch(color=pareto_colors_plot[i], label=pareto_names[i]) for i in range(len(pareto_names))]
    ax.legend(handles=handles, fontsize=9, loc='upper right', title='Pareto Points')
    
    ax.set_xlabel(PARAM_NAMES[0], fontsize=11)
    ax.set_ylabel(PARAM_NAMES[1], fontsize=11)
    ax.set_title('Solution Heat Treatment', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Second Heat Treatment (T2 vs t2)
    ax = axs[0, 1]
    ax.scatter(train_x[:, 2], train_x[:, 3], color='blue', alpha=0.6, s=80,
               edgecolor='black', linewidth=0.5, label='All experiments')
    ax.scatter(pareto_x[:, 2], pareto_x[:, 3], color=pareto_colors_plot, s=120,
               marker='X', edgecolor=pareto_colors_plot, label='Pareto optimal', zorder=10)
    ax.set_xlabel(PARAM_NAMES[2], fontsize=11)
    ax.set_ylabel(PARAM_NAMES[3], fontsize=11)
    ax.set_title('Aging Treatment', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Temperature Distribution (T1 vs T2)
    ax = axs[1, 0]
    ax.scatter(train_x[:, 0], train_x[:, 2], color='blue', alpha=0.6, s=80,
               edgecolor='black', linewidth=0.5, label='All experiments')
    ax.scatter(pareto_x[:, 0], pareto_x[:, 2], color=pareto_colors_plot, s=120,
               marker='X', edgecolor=pareto_colors_plot, label='Pareto optimal', zorder=10)
    
    ax.set_xlabel(PARAM_NAMES[0], fontsize=11)
    ax.set_ylabel(PARAM_NAMES[2], fontsize=11)
    ax.set_title('Temperature Distribution', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Time Distribution (t1 vs t2)
    ax = axs[1, 1]
    ax.scatter(train_x[:, 1], train_x[:, 3], color='blue', alpha=0.6, s=80,
               edgecolor='black', linewidth=0.5, label='All experiments')
    ax.scatter(pareto_x[:, 1], pareto_x[:, 3], color=pareto_colors_plot, s=120,
               marker='X', edgecolor=pareto_colors_plot, label='Pareto optimal', zorder=10)
    ax.set_xlabel(PARAM_NAMES[1], fontsize=11)
    ax.set_ylabel(PARAM_NAMES[3], fontsize=11)
    ax.set_title('Time Distribution', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(METRIC_FIGURES_PATH, 'experiment_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Generated: {METRIC_FIGURES_PATH}/experiment_distribution.png")

if __name__ == "__main__":

    c_blind_5254 = [
        "#333333",
        "#D55E00",
        "#0072B2",
        "#66A61E",
        "#A6761D",
        "#E7298A"
    ]

    train_x, train_y_transformed, train_y_original = load_data_from_database()
    pareto_x, pareto_y, pareto_y_original = bo.identify_pareto_front(train_x, train_y_transformed, train_y_original)
    plot_parameter_objective_correlations(train_x, train_y_original, pareto_x, pareto_y_original, c_blind_5254)
    plot_parameter_importance_summary(train_x, train_y_original)
    loocv_validation_detailed(train_x, train_y_transformed, train_y_original, bo.BOUNDS)
    uncertainty_heatmaps(train_x, train_y_transformed, train_y_original, bo.BOUNDS, c_blind_5254)
    plot_experiment_distribution(train_x, train_y_transformed, train_y_original,c_blind_5254)
