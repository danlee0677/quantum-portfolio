import json
import matplotlib.pyplot as plt
max_limit = 6
budget_diff_penalizer = 1


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

def compare_solutions(data, data2, data3, data4):
    results = {}
    
    for case_id, case_data in data.items():
        n_qubits = case_data["hyperparams"]["n_qubits"]
        cont_sol = data3[case_id]["continuous_variables_solution"]
        exact_sol = data4[case_id]["exact_solution"]
        qaoa_sol = case_data["qaoa_solution"]
        budget = case_data["hyperparams"]["budget"]
        un_cont_sol = data2[case_id]["continuous_variables_solution_unconstrained"]
        
        allocation2 = un_cont_sol["allocation"]
        used_budget = 0
        for stock, weight in allocation2.items():
            used_budget += weight * data2[case_id]["hyperparams"]["prices_now"][stock]
        unconstrained_cont_value = un_cont_sol["value"]
        unconstrained_cont_penalty = budget_diff_penalizer*un_cont_sol["left_overs"]
        
        cont_value = cont_sol["value"]
        cont_penalty = budget_diff_penalizer*cont_sol["left_overs"]

        exact_budgets = exact_sol.get("result_with_budget", [])[0]
        exact_value = exact_budgets.get("objective_value", 0)
        exact_penalty = budget_diff_penalizer*exact_budgets["difference"]

        qaoa_value = qaoa_sol.get("objective_values", [])[-1]
        qaoa_budgets = qaoa_sol.get("result_with_budget", [])[-1]
        qaoa_penalty = budget_diff_penalizer*qaoa_budgets["difference"]

        results[case_id] = {
            "continuous": {"value": cont_value, "penalty": cont_penalty},
            "continuous_unconstrained": {"value": unconstrained_cont_value, "penalty": unconstrained_cont_penalty},
            "exact": {"value": exact_value, "penalty": exact_penalty},
            "qaoa": {"value": qaoa_value, "penalty": qaoa_penalty},
            "n_qubits": n_qubits,
            "budget": budget
        }
    
    return results


def plot_objective_vs_budget(results, save_path=None, dpi=300, fig_height=7):
    """
    Creates a publication-quality scatter plot comparing objective values vs budget utilization,
    with a magnified inset showing the upper right corner.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the results for each case and method.
    save_path : str, optional
        Path to save the figure. If None, the figure is displayed but not saved.
    dpi : int, optional
        Resolution of the saved figure in dots per inch.
    fig_width : float, optional
        Width of the figure in inches.
    fig_height : float, optional
        Height of the figure in inches.
    
    Returns:
    --------
    fig, ax : tuple
        Figure and axes objects for further customization if needed.
    """
    fig_width = 1.618 * fig_height  # Golden ratio for figure width
    # Set publication-quality parameters
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 16,
        'axes.linewidth': 0.8,
        'axes.labelsize': 16,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 14,
        'legend.frameon': False,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.5
    })
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Set smoke gray background for plotting area
    ax.set_facecolor('#f0f0f0')  # Light smoke gray
    
    # Define a professional color palette (colorblind-friendly)
    colors = ['#0173B2', '#DE8F05', '#029E73']
    markers = ['o', 's', '^', 'P', 'D']
    
    # Extract data for plotting
    continuous_obj_values = []
    exact_obj_values = []
    qaoa_obj_values = []
    unconstrained_obj_values = []
    
    continuous_budget_pct = []
    exact_budget_pct = []
    qaoa_budget_pct = []
    unconstrained_budget_pct = []
    
    case_labels = []
    
    # Process data for plotting
    for case_id, case_data in results.items():
        case_labels.append(case_id)

        vals = [case_data["continuous"]["value"], case_data["exact"]["value"], case_data["qaoa"]["value"], case_data["continuous_unconstrained"]["value"]]
        max_val = max(vals)
        min_val = min(vals)
        obj_range = max_val - min_val
        # Normalize objective values to [0,1] scale
        cont_obj = (case_data["continuous"]["value"] - min_val)/obj_range
        exact_obj = (case_data["exact"]["value"]- min_val)/obj_range
        qaoa_obj = (case_data["qaoa"]["value"]- min_val)/obj_range
        unconstrained_cont_obj = (case_data["continuous_unconstrained"]["value"]- min_val)/obj_range
        
        if case_id == "47":
            print(f"Case {case_id} - cont_obj: {case_data['continuous']['value']}, exact_obj: {case_data['exact']['value']}, qaoa_obj: {case_data['qaoa']['value']}, unconstrained_cont_obj: {case_data['continuous_unconstrained']['value']}")

        continuous_obj_values.append(cont_obj)
        exact_obj_values.append(exact_obj)
        qaoa_obj_values.append(qaoa_obj)
        unconstrained_obj_values.append(unconstrained_cont_obj)
        
        # Calculate budget utilization percentage
        budget = case_data["budget"]
        cont_budget_pct = 100 - case_data["continuous"]["penalty"] / budget * 100
        ex_budget_pct = 100 - case_data["exact"]["penalty"] / budget * 100
        q_budget_pct = 100 - case_data["qaoa"]["penalty"] / budget * 100
        u_budget_pct = 100 - case_data["continuous_unconstrained"]["penalty"] / budget * 100
        
        continuous_budget_pct.append(cont_budget_pct)
        exact_budget_pct.append(ex_budget_pct)
        qaoa_budget_pct.append(q_budget_pct)
        unconstrained_budget_pct.append(u_budget_pct)
    
    # Plot scatter points
    cont_scatter = ax.scatter(continuous_obj_values, continuous_budget_pct, 
                             label="Cont. constrained + discretization", 
                             color=colors[0], marker=markers[0], s=300, 
                             edgecolor='black', linewidth=0.5, alpha=0.8)
    
    unconstrained_scatter = ax.scatter(unconstrained_obj_values, unconstrained_budget_pct, 
                                       label="Cont. unconstrained + discretization", 
                                       color='gray', marker=markers[3], s=300, 
                                       edgecolor='black', linewidth=0.5, alpha=0.8)
    
    exact_scatter = ax.scatter(exact_obj_values, exact_budget_pct, 
                              label="Eigensolver sol. to HUBO", 
                              color=colors[1], marker=markers[1], s=300, 
                              edgecolor='black', linewidth=0.5, alpha=0.8)
    
    qaoa_scatter = ax.scatter(qaoa_obj_values, qaoa_budget_pct, 
                             label="QAOA sol. to HUBO", 
                             color=colors[2], marker=markers[2], s=300, 
                             edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add text labels for cases
    for i, case in enumerate(case_labels):
        font_size = 16
        xytext = (6, 6)  # Offset for text labels
        ax.annotate(case, (continuous_obj_values[i], continuous_budget_pct[i]),
                   xytext=xytext, textcoords='offset points', fontsize=font_size)
        ax.annotate(case, (exact_obj_values[i], exact_budget_pct[i]),
                   xytext=xytext, textcoords='offset points', fontsize=font_size)
        ax.annotate(case, (qaoa_obj_values[i], qaoa_budget_pct[i]),
                   xytext=xytext, textcoords='offset points', fontsize=font_size)
        ax.annotate(case, (unconstrained_obj_values[i], unconstrained_budget_pct[i]),
                    xytext=xytext, textcoords='offset points', fontsize=font_size)
    
    # Add reference lines
    ax.axhline(y=100, color='black', linestyle='--', alpha=0.3, zorder=0)
    ax.axvline(x=1, color='black', linestyle='--', alpha=0.3, zorder=0)
    
    # Customize axes
    ax.set_xlabel("Norm. objective, no budget constraint (higher is better)", fontweight='bold', fontsize=24)
    ax.set_ylabel("Budget utilization (%)", fontweight='bold', fontsize=24)
    
    # Set axis limits with some padding
    min_obj = min(min(continuous_obj_values), min(exact_obj_values), min(qaoa_obj_values), min(unconstrained_obj_values))
    max_obj = max(max(continuous_obj_values), max(exact_obj_values), max(qaoa_obj_values), max(unconstrained_obj_values))
    obj_margin = (max_obj - min_obj) * 0.1
    ax.set_xlim(min_obj - obj_margin, max_obj + obj_margin)

    min_budget = min(min(continuous_budget_pct), min(exact_budget_pct), min(qaoa_budget_pct), min(unconstrained_budget_pct))
    max_budget = max(max(continuous_budget_pct), max(exact_budget_pct), max(qaoa_budget_pct), max(unconstrained_budget_pct))
    budget_margin = (max_budget - min_budget) * 0.1
    ax.set_ylim(min_budget - budget_margin, max_budget + budget_margin)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    
    # Customize spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend in optimal position
    ax.legend(loc='lower right', ncol=2, bbox_to_anchor=(0.9, 0.05), fontsize=16)
    
    # Adjust layout
    #plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig, ax