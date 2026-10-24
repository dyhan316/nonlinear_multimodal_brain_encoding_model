import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# Try importing adjustText
try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("⚠️ 'adjustText' not detected. Install via `pip install adjustText`.")

# ==========================================
# 1. Setup & Configuration
# ==========================================

# plt.rcParams['font.size'] = 22          # default text
# plt.rcParams['axes.titlesize'] = 26     # subplot titles
# plt.rcParams['axes.labelsize'] = 26     # x/y label sizes
# plt.rcParams['xtick.labelsize'] = 20    # tick labels
# plt.rcParams['ytick.labelsize'] = 20
# plt.rcParams['legend.fontsize'] = 22    # legend text
# plt.rcParams['figure.titlesize'] = 28   # figure title
# plt.rcParams['figure.figsize'] = (14, 10)  # default size

plt.rcParams.update({
    "font.size": 18,       # base font size
    "axes.labelsize": 20,  # x/y labels
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 22,
})

OUTPUT_DIR = 'results_images_lucy_clean'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📂 Created output directory: {OUTPUT_DIR}")

# Updated Display Names: "Model Name Size (Training Data)"
model2name_dict = {
    # LLaMA (Standard)
    "LLAMA7B": "LLaMA 7B",
    "LLAMA13B": "LLaMA 13B",
    "LLAMA30B": "LLaMA 33B",
    "LLAMA65B": "LLaMA 65B",
    
    # Pythia (Standard)
    "EleutherAI/pythia-14m": "Pythia 14M",
    "EleutherAI/pythia-70m": "Pythia 70M",
    "EleutherAI/pythia-160m": "Pythia 160M",
    
    # BabyLM (10M Data Track)
    "babylm-babyllama-10m-2024": "BabyLLaMA 58M (10M)",
    "babylm/opt-125m-strict-small-2023": "OPT 125M (10M)",
    "babylm/ltgbert-10m-2024": "LTGBERT BASE (10M)",
    "babylm/t5-base-strict-small-2023": "T5 Base 220M (10M)",
    
    # BabyLM (100M Data Track)
    "babylm/babyllama-100m-2024": "BabyLLaMA 58M (100M)",
    "babylm/opt-125m-strict-2023": "OPT 125M (100M)",
    "babylm/ltgbert-100m-2024": "LTGBERT BASE (100M)",
    "babylm/t5-base-strict-2023": "T5 Base 220M (100M)",
}

MODEL_SIZES = {
    "LLAMA7B": 7000, "LLAMA13B": 13000, "LLAMA30B": 33000, "LLAMA65B": 65000,
    "babylm-babyllama-10m-2024": 58, "babylm/babyllama-100m-2024": 58,
    "EleutherAI/pythia-14m": 14, "EleutherAI/pythia-70m": 70, "EleutherAI/pythia-160m": 160,
    "babylm/opt-125m-strict-2023": 125, "babylm/opt-125m-strict-small-2023": 125,
    "babylm/ltgbert-100m-2024": 100, "babylm/ltgbert-10m-2024": 100,
    "babylm/t5-base-strict-2023": 220, "babylm/t5-base-strict-small-2023": 220,
}

NUM_LAYERS_DICT = {
    'EleutherAI/pythia-14m': 6, 'EleutherAI/pythia-160m': 12, 'EleutherAI/pythia-70m': 6,
    'LLAMA13B': 40, 'LLAMA27Bhf': 32, 'LLAMA30B': 60, 'LLAMA38B': 32, 'LLAMA65B': 80, 'LLAMA7B': 32,
    'babylm-babyllama-10m-2024': 16, 'babylm/babyllama-100m-2024': 16,
    'babylm/ltgbert-100m-2024': 12, 'babylm/ltgbert-10m-2024': 12,
    'babylm/opt-125m-strict-2023': 12, 'babylm/opt-125m-strict-small-2023': 12,
    'babylm/t5-base-strict-2023': 12, 'babylm/t5-base-strict-small-2023': 12
}

# Groups
llama_models = ["LLAMA7B", "LLAMA13B", "LLAMA30B", "LLAMA65B"]
pythia_models = ["EleutherAI/pythia-14m", "EleutherAI/pythia-70m", "EleutherAI/pythia-160m"]
babylm_10m = ["babylm-babyllama-10m-2024", "babylm/opt-125m-strict-small-2023", "babylm/ltgbert-10m-2024", "babylm/t5-base-strict-small-2023"]
babylm_100m = ["babylm/babyllama-100m-2024", "babylm/opt-125m-strict-2023", "babylm/ltgbert-100m-2024", "babylm/t5-base-strict-2023"]

STYLES = {
    "LLAMA7B": {"color": "#4eb3d3", "marker": "o"},
    "LLAMA13B": {"color": "#2b8cbe", "marker": "^"},
    "LLAMA30B": {"color": "#0868ac", "marker": "s"},
    "LLAMA65B": {"color": "#084081", "marker": "D"},
    "EleutherAI/pythia-14m": {"color": "#fdbb84", "marker": "o"},
    "EleutherAI/pythia-70m": {"color": "#e34a33", "marker": "^"},
    "EleutherAI/pythia-160m": {"color": "#b30000", "marker": "s"},
    "babylm-babyllama-10m-2024": {"color": "tab:purple", "marker": "o"},
    "babylm/babyllama-100m-2024": {"color": "tab:purple", "marker": "o"},
    "babylm/opt-125m-strict-small-2023": {"color": "tab:gray", "marker": "v"},
    "babylm/opt-125m-strict-2023": {"color": "tab:gray", "marker": "v"},
    "babylm/ltgbert-10m-2024": {"color": "tab:blue", "marker": "s"},
    "babylm/ltgbert-100m-2024": {"color": "tab:blue", "marker": "s"},
    "babylm/t5-base-strict-small-2023": {"color": "tab:green", "marker": "D"},
    "babylm/t5-base-strict-2023": {"color": "tab:green", "marker": "D"},
}

# STYLES = {
    
#     "LLAMA7B": {"color": "#4eb3d3", "marker": "o"},
#     "LLAMA13B": {"color": "#2b8cbe", "marker": "^"},
#     "LLAMA30B": {"color": "#0868ac", "marker": "s"},
#     "LLAMA65B": {"color": "#084081", "marker": "D"},
#     "EleutherAI/pythia-14m": {"color": "#fdbb84", "marker": "o"},
#     "EleutherAI/pythia-70m": {"color": "#e34a33", "marker": "^"},
#     "EleutherAI/pythia-160m": {"color": "#b30000", "marker": "s"},
#     "babylm-babyllama-10m-2024": {"color": "tab:purple", "marker": "o"},
#     "babylm/babyllama-100m-2024": {"color": "#c5b0d5", "marker": "o"},
#     "babylm/opt-125m-strict-small-2023": {"color": "lightgray", "marker": "v"},
#     "babylm/opt-125m-strict-2023": {"color": "tab:gray", "marker": "v"},
#     "babylm/ltgbert-10m-2024": {"color": "tab:cyan", "marker": "s"},
#     "babylm/ltgbert-100m-2024": {"color": "#1b9e9e", "marker": "s"},
#     "babylm/t5-base-strict-small-2023": {"color": "#98df8a", "marker": "D"},
#     "babylm/t5-base-strict-2023": {"color": "tab:green", "marker": "D"},
# }

# ==========================================
# 2. Data Processing & Smoothing
# ==========================================

def load_data():
    print("⏳ Loading data...")
    try:
        df_prev = pd.read_csv('all_results_old_and_new.csv')
        df_curr = pd.read_csv('all_results_1207.csv')
    except FileNotFoundError:
        print("❌ Error: CSV files not found.")
        return None

    df_babylm_10m = df_prev[df_prev.exp_info.str.startswith('babylm') & df_prev.encoding_setting.str.contains('default')]
    df_llama = df_prev[df_prev.exp_info.str.startswith('LLAMA') & df_prev.encoding_setting.str.contains('default')]
    df_curr_clean = df_curr[~df_curr.exp_info.str.startswith(('llama', 'babyllama-10m'))]
    
    df_all = pd.concat([df_llama, df_curr_clean, df_babylm_10m], ignore_index=True)
    df_final = df_all[(df_all['encoding_setting'] == 'default') & (df_all['response_setting'] == 'default')].copy()

    def parse(s):
        m = re.search(r'^(.*)_(\d+)$', s)
        return (m.group(1), int(m.group(2))) if m else (None, None)

    df_final[['model_name', 'layer_num']] = df_final['stimulus_setting'].apply(lambda x: pd.Series(parse(x)))
    df_final.dropna(subset=['model_name', 'layer_num'], inplace=True)
    
    # === CRITICAL FIX: Pre-Calculate Smoothed Max for Every Model ===
    # This ensures consistency across all plots. We replace the raw score with the smoothed score.
    models_to_smooth = llama_models # Only apply aggressive despiking to LLaMA as requested
    
    # We will iterate through groups and apply smoothing where necessary
    # Note: For this script, I will apply the `remove_spikes` logic to LLaMA models 
    # directly in the plotting functions or by modifying the DF here. 
    # Modifying the DF here is safer for consistency.
    
    return df_final

def get_smoothed_series(df, model, despike=False):
    """Returns the smoothed/despiked series for a model."""
    sub = df[df['model_name'] == model].sort_values('layer_num').copy()
    if sub.empty: return None
    
    if despike:
        # Robust Spike Removal
        col = 'mean_r2_mean'
        rolling_med = sub[col].rolling(window=5, center=True, min_periods=1).median()
        is_spike = sub[col] > (rolling_med * 1.10) # 10% threshold
        sub.loc[is_spike, col] = np.nan
        sub[col] = sub[col].interpolate(method='linear')
        
    return sub

# ==========================================
# 3. Plotting Functions
# ==========================================

def save_legend(ax, filename, ncol=1, legend_fontsize=20):
    """Save legend from a given axes as a separate PNG."""
    handles, labels = ax.get_legend_handles_labels()

    fig_legend = plt.figure(figsize=(4, 2))  # adjust size as needed
    fig_legend.legend(
        handles, labels,
        loc='center',
        ncol=ncol,
        fontsize=legend_fontsize,
        frameon=False,
    )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig_legend.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig_legend)


def plot_layer_trajectories(df):
    """Figure 1"""
    print("   ... Generating Layer Trajectories")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    def plot_sub(ax, models, title, ymax=None, despike=False):
        for m in models:
            sub = get_smoothed_series(df, m, despike=despike)
            if sub is None: continue

            total = NUM_LAYERS_DICT.get(m, sub['layer_num'].max())
            x = sub['layer_num'] / (total if total > 0 else 1)
            
            style = STYLES.get(m, {"color": "black", "marker": "."})
            
            ax.plot(x, sub['mean_r2_mean'], label=model2name_dict.get(m, m), 
                    color=style["color"], marker=style["marker"], 
                    linewidth=2, markersize=6, alpha=0.8)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if ymax: ax.set_ylim(0, ymax)
        #ax.legend(fontsize=22, loc='upper left')

    plot_sub(axs[0, 0], llama_models, "(a) LLaMA Scaling (Smoothed)", despike=True)
    plot_sub(axs[0, 1], pythia_models, "(b) Pythia Scaling")
    plot_sub(axs[1, 0], babylm_10m, "(c) BabyLM (10M Data)")
    plot_sub(axs[1, 1], babylm_100m, "(d) BabyLM (100M Data)")
    
    save_legend(axs[0, 0], "results_images_lucy_clean/legends/a_llama_scaling.png", ncol=1)
    save_legend(axs[0, 1], "results_images_lucy_clean/legends/b_pythia_scaling.png", ncol=1)
    save_legend(axs[1, 0], "results_images_lucy_clean/legends/c_babylm_10m.png", ncol=1)
    save_legend(axs[1, 1], "results_images_lucy_clean/legends/d_babylm_100m.png", ncol=1)

    fig.text(0.5, 0.02, 'Normalized Layer Depth', ha='center', fontsize=14)
    fig.text(0.02, 0.5, 'Encoder Performance ($R^2$)', va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.03, 0.03, 1, 1])
    plt.savefig(f"{OUTPUT_DIR}/1_layer_trajectories.png", dpi=300)
    plt.close()

def plot_peak_performance(df):
    """Figure 2: Peak Performance Bar Chart (All Models)"""
    print("   ... Generating Peak Performance Bar Chart (All Models)")
    
    # Calculate Peaks
    peak_data = {}
    
    # --- CHANGE: Get ALL unique models from the dataframe instead of a subset ---
    #df = df[~df['model_name'].str.startswith("LLAMA", na = False)]
    models_to_exclude = ['LLAMA38B', 'LLAMA27Bhf']
    df = df[~df['model_name'].isin(models_to_exclude)]
    all_models = df['model_name'].unique()
    
    for m in all_models:
        # Despike LLaMA only, keep others raw
        # Ensure 'llama_models' list is defined globally or passed in
        despike_flag = True if m in llama_models else False
        
        sub = get_smoothed_series(df, m, despike=despike_flag)
        
        if sub is not None:
            max_score = sub['mean_r2_mean'].max()
            peak_data[m] = max_score

    # Sort by score (Highest at top)
    sorted_peaks = dict(sorted(peak_data.items(), key=lambda item: item[1]))
    
    plt.figure(figsize=(14, 10))  # Increased height slightly to fit more bars
    colors = []
    labels = []
    values = []
    
    for m, score in sorted_peaks.items():
        # Get display name (fallback to raw name if missing)
        labels.append(model2name_dict.get(m, m))
        values.append(score)
        
        # Color Logic
        #colors.append(STYLES.get(m, {"color": "gray"})["color"])
        if "LLAMA" in m: colors.append(STYLES["LLAMA65B"]["color"])
        elif "pythia" in m: colors.append(STYLES["EleutherAI/pythia-160m"]["color"])
        elif "ltgbert" in m: colors.append(STYLES["babylm/ltgbert-100m-2024"]["color"])
        elif "t5" in m: colors.append(STYLES["babylm/t5-base-strict-2023"]["color"])
        elif "opt" in m: colors.append(STYLES["babylm/opt-125m-strict-2023"]["color"]) # Added OPT color
        elif "babyllama" in m: colors.append(STYLES["babylm-babyllama-10m-2024"]["color"]) # Added BabyLLaMA color
        else: colors.append("gray")

    bars = plt.barh(labels, values, color=colors, alpha=0.7)
    plt.xlim(0, max(values) * 1.10)
    plt.xlabel("Peak Encoding Performance ($R^2$)", fontsize=12)
    plt.title("Best Layer Performance (All Models)", fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    
    # Add values to bars
    max_val = max(values) if values else 0
    for bar in bars:
        width = bar.get_width()
        plt.text(width + (max_val * 0.010), bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center', fontsize=15, ha='left')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_peak_performance.png", dpi=300)
    plt.close()

#def plot_scaling_law(df):
    """Figure 3: Scaling Law (Using Smoothed Values + All Labels)"""
    print("   ... Generating Scaling Law Plot")
    
    plt.figure(figsize=(9, 7))
    texts = []
    
    for m, size in MODEL_SIZES.items():
        # Despike LLaMA
        despike_flag = True if m in llama_models else False
        sub = get_smoothed_series(df, m, despike=despike_flag)
        
        if sub is not None:
            score = sub['mean_r2_mean'].max()
            
            style = STYLES.get(m, {"color": "gray"})
            plt.scatter(size, score, color=style["color"], s=150, alpha=0.8, edgecolors='white', zorder=2)
            
            # Label Every Model
            label_text = model2name_dict.get(m, m)
            texts.append(plt.text(size, score, label_text, fontsize=15))

    plt.xscale('log')
    plt.xlabel("Model Parameters (Millions)", fontsize=12)
    plt.ylabel("Peak Encoding Performance ($R^2$)", fontsize=12)
    plt.title("Scaling Law: Brain Alignment vs. Model Size", fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    if HAS_ADJUST_TEXT:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_scaling_law.png", dpi=300)
    plt.close()

def plot_scaling_law(df):
    print("   ... Generating Scaling Law Plot")
    
    plt.figure(figsize=(9, 7))
    texts = []
    
    for m, size in MODEL_SIZES.items():
        despike_flag = True if m in llama_models else False
        sub = get_smoothed_series(df, m, despike=despike_flag)
        
        if sub is not None:
            score = sub['mean_r2_mean'].max()
            style = STYLES.get(m, {"color": "gray"})
            
            # --- ADD JITTER FOR CROWDED MODELS ---
            if m in llama_models:
                jitter_x = size * (1 + np.random.uniform(-0.05, 0.05))
                jitter_y = score * (1 + np.random.uniform(-0.03, 0.03))
            else:
                jitter_x, jitter_y = size, score
            
            plt.scatter(
                jitter_x, jitter_y,
                color=style["color"],
                s=150, alpha=0.8,
                edgecolors='white', zorder=2
            )
            
            label_text = model2name_dict.get(m, m)
            texts.append(
                plt.text(jitter_x, jitter_y, label_text, fontsize=15)
            )

    plt.xscale('log')
    plt.xlabel("Model Parameters (Millions)", fontsize=14)
    plt.ylabel("Peak Encoding Performance ($R^2$)", fontsize=14)
    plt.title("Scaling Law: Brain Alignment vs. Model Size", fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.4)

    if HAS_ADJUST_TEXT:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', lw=0.5, color='gray'))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_scaling_law.png", dpi=300)
    plt.close()


def plot_data_efficiency(df):
    """Figure 4: Data Efficiency"""
    print("   ... Generating Data Efficiency Plot")
    pairs = [
        ("BabyLLaMA", "babylm-babyllama-10m-2024", "babylm/babyllama-100m-2024"),
        ("OPT", "babylm/opt-125m-strict-small-2023", "babylm/opt-125m-strict-2023"),
        ("LTGBERT", "babylm/ltgbert-10m-2024", "babylm/ltgbert-100m-2024"),
        ("T5", "babylm/t5-base-strict-small-2023", "babylm/t5-base-strict-2023")
    ]
    
    labels = []
    s10m, s100m = [], []
    
    for name, m10, m100 in pairs:
        # No despiking needed for small models usually, but consistent to check
        sub10 = get_smoothed_series(df, m10)
        sub100 = get_smoothed_series(df, m100)
        
        if sub10 is not None and sub100 is not None:
            labels.append(name)
            s10m.append(sub10['mean_r2_mean'].max())
            s100m.append(sub100['mean_r2_mean'].max())
            
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(9, 6))
    plt.bar(x - width/2, s10m, width, label='10M Tokens', color='#ff9999', alpha=0.9)
    plt.bar(x + width/2, s100m, width, label='100M Tokens', color='#66b3ff', alpha=0.9)
    plt.ylabel('Peak $R^2$ Score')
    plt.title('Impact of Training Data Size (10x Data Increase)')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_data_efficiency.png", dpi=300)
    plt.close()

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        #plot_layer_trajectories(df)
        plot_peak_performance(df)
        #plot_scaling_law(df)
        #plot_data_efficiency(df)
        print(f"✅ All plots saved to '{OUTPUT_DIR}/'")