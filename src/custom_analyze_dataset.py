import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

def compare_variable_distributions(data, oc, outcome_col="Sepsis3",
                                   min_samples=100, plot=True):
    """
    Compare distributions of each variable between outcome groups.
    Assumes `data` is long-format with columns:
      ['ts_id', 'variable', 'value', 'minute']
    """

    # Merge outcome into data
    merged = data.merge(
        oc[['ts_id', outcome_col]],
        on='ts_id',
        how='inner'
    )

    variables = merged['variable'].unique()

    print("\n=== Variable Distribution Comparison ===")

    results = []

    for var in variables:
        var_df = merged[merged['variable'] == var]

        pos_vals = var_df[var_df[outcome_col] == 1]['value'].dropna()
        neg_vals = var_df[var_df[outcome_col] == 0]['value'].dropna()

        if len(pos_vals) < min_samples or len(neg_vals) < min_samples:
            continue

        # Summary stats
        pos_mean, neg_mean = pos_vals.mean(), neg_vals.mean()
        pos_std, neg_std = pos_vals.std(), neg_vals.std()

        # KS test
        ks_stat, ks_p = ks_2samp(pos_vals, neg_vals)

        results.append({
            "variable": var,
            "n_pos": len(pos_vals),
            "n_neg": len(neg_vals),
            "mean_pos": pos_mean,
            "mean_neg": neg_mean,
            "std_pos": pos_std,
            "std_neg": neg_std,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p
        })

        print(
            f"{var:25s} | "
            f"mean(pos)={pos_mean:.3f}, mean(neg)={neg_mean:.3f} | "
            f"KS p={ks_p:.2e}"
        )

        # Plot distributions
        if plot:
            # Robust x-axis limits (ignore extreme outliers for visualization)
            all_vals = np.concatenate([neg_vals.values, pos_vals.values])
            x_min, x_max = np.percentile(all_vals, [1, 99])

            plt.figure()
            plt.hist(
                neg_vals,
                bins=50,
                range=(x_min, x_max),
                alpha=0.6,
                density=True,
                label="Negative"
            )
            plt.hist(
                pos_vals,
                bins=50,
                range=(x_min, x_max),
                alpha=0.6,
                density=True,
                label="Positive"
            )

            plt.xlim(x_min, x_max)
            plt.title(f"{var} distribution by outcome")
            plt.xlabel(var)
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.show()


    return results



def main(pkl_path):
    # Load the dataset pickle
    with open(pkl_path, 'rb') as f:
        data, oc, train_ids, val_ids, test_ids, infer_ids = pickle.load(f)

        print(oc.head(20))
        print(data.head(20))

    print("=== Dataset Split Info ===")
    print(f"Total TS in pickle: {len(np.unique(data['ts_id']))}")
    print(f"# Train IDs: {len(train_ids)}")
    print(f"# Validation IDs: {len(val_ids)}")
    print(f"# Test IDs: {len(test_ids)}")
    print(f"# Infer IDs: {len(infer_ids)}")

    # Check for overlap
    print("\n=== Overlap Checks ===")
    print(f"Train ∩ Val: {len(set(train_ids) & set(val_ids))}")
    print(f"Train ∩ Test: {len(set(train_ids) & set(test_ids))}")
    print(f"Train ∩ Infer: {len(set(train_ids) & set(infer_ids))}")
    print(f"Val ∩ Test: {len(set(val_ids) & set(test_ids))}")
    print(f"Val ∩ Infer: {len(set(val_ids) & set(infer_ids))}")
    print(f"Test ∩ Infer: {len(set(test_ids) & set(infer_ids))}")

    # Check distribution of the outcome
    print("\n=== Outcome Distribution ===")
    ts_id_to_ind = {ts_id:i for i, ts_id in enumerate(np.unique(data['ts_id']))}
    
    def get_outcome_stats(ids, label_name='Sepsis3'):
        inds = [ts_id_to_ind[i] for i in ids if i in ts_id_to_ind]
        labels = oc.loc[oc['ts_id'].isin(ids), label_name]
        return len(labels), labels.sum(), labels.mean()
    
    for split_name, split_ids in zip(['Train','Val','Test','Infer'],
                                     [train_ids, val_ids, test_ids, infer_ids]):
        n, pos, frac = get_outcome_stats(split_ids)
        print(f"{split_name}: {n} samples, {pos} positive, {frac:.3f} positive fraction")

    # Check time coverage per TS
    print("\n=== Time Coverage per TS ===")
    for split_name, split_ids in zip(['Train','Val','Test','Infer'],
                                     [train_ids, val_ids, test_ids, infer_ids]):
        ts_data = data[data['ts_id'].isin(split_ids)]
        ts_minute_range = ts_data.groupby('ts_id')['minute'].agg(['min','max'])
        print(f"{split_name}: min time = {ts_minute_range['min'].min()}, max time = {ts_minute_range['max'].max()}")

    # --- New: Show variables in dataset ---
    all_variables = data['variable'].unique()
    print("\n=== Variables in Dataset ===")
    print(f"Total variables: {len(all_variables)}")
    print(all_variables)

    # === Compare variable distributions by outcome ===
    compare_variable_distributions(
        data,
        oc,
        outcome_col="Sepsis3",
        min_samples=500,
        plot=True
    )

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True,
                        help="Path to the dataset pickle file")
    args = parser.parse_args()
    main(args.pkl_path)

#  python help.py --pkl_path C:\Users\Skyfinder\Projects\STraTS\data\processed\aumc-4.pkl