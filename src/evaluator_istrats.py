from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, brier_score_loss, precision_score
from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt
import os

class Evaluator:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def expected_calibration_error(y_true, y_prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins) - 1
        ece = 0.0
        for i in range(n_bins):
            mask = bin_ids == i
            if mask.sum() > 0:
                acc = y_true[mask].mean()
                conf = y_prob[mask].mean()
                ece += np.abs(acc - conf) * mask.mean()
        return ece

    def evaluate(self, model, dataset, split, train_step):
        self.args.logger.write('\nEvaluating on split = ' + split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0, num_samples, self.args.eval_batch_size),
                    desc='running forward pass')
        true, pred = [], []

        # For interpretable aggregation
        all_obs_contrib = []
        all_att = []
        all_varis = []
        all_times = []
        all_values = []

        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            labels = batch['labels']
            true.append(labels)

            # Move batch to device
            batch_device = {k: v.to(self.args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                if split == 'infer':
                    out = model(**batch_device, return_interpret=True)
                    logits = out['logits'].cpu()
                    pred.append(torch.sigmoid(logits))

                    # Only collect interpretable outputs if they exist
                    if out['obs_contrib'] is not None:
                        obs_contrib = out['obs_contrib'].cpu()     # (B, T)
                        att = out['att_weights'].cpu()             # (B, T)
                        varis = batch['varis'].cpu()               # (B, T)
                        times = batch['times'].cpu()
                        values = batch['values'].cpu()

                        # Valid observation mask (must match model logic)
                        obs_mask = (values != 0)

                        # Flatten only valid observations
                        all_obs_contrib.append(obs_contrib[obs_mask])
                        all_att.append(att[obs_mask])
                        all_varis.append(varis[obs_mask])
                        all_times.append(times[obs_mask])
                        all_values.append(values[obs_mask])

                else:
                    # Non-inference mode
                    logits_or_loss = model(**batch_device)
                    if labels is not None:
                        # Run without labels to get logits for metrics
                        batch_nolab = {**batch_device}
                        batch_nolab.pop('labels', None)
                        logits = model(**batch_nolab)
                    else:
                        logits = logits_or_loss
                    pred.append(logits.cpu())

        # Concatenate true labels and predictions
        true = torch.cat(true).cpu().numpy()
        pred = torch.cat(pred).cpu().numpy()

        # ======================
        # Core metrics
        # ======================
        precision, recall, _ = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true, pred)
        brier = brier_score_loss(true, pred)
        ece = self.expected_calibration_error(true, pred)
        pred_labels = (pred >= 0.5).astype(int)
        ppv = precision_score(true, pred_labels)

        result = {
            'auroc': roc_auc,
            'auprc': pr_auc,
            'minrp': minrp,
            'brier': brier,
            'ece': ece,
            'ppv': ppv
        }

        if train_step is not None:
            self.args.logger.write(
                f"Result on {split} split at train step {train_step}: {result}"
            )

        # ======================
        # Inference split: plot ROC/PR, histograms, calibration, save interpretable outputs
        # ======================
        if split == 'infer' and self.args.model_type == 'istrats':
            if all_obs_contrib:  # only if collected
                obs_contrib = torch.cat(all_obs_contrib, dim=0)
                att = torch.cat(all_att, dim=0)
                varis = torch.cat(all_varis, dim=0)
                times = torch.cat(all_times, dim=0)
                values = torch.cat(all_values, dim=0)

                # Create mask for valid observations (non-padding)
                obs_mask = (values != 0)

                # Compute per-variable mean contribution
                num_vars = self.args.V
                var_contrib = torch.zeros(num_vars)
                counts = torch.zeros(num_vars)

                for v in range(num_vars):
                    mask_v = obs_mask & (varis == v)
                    if mask_v.any():
                        var_contrib[v] = obs_contrib[mask_v].mean()
                        counts[v] = mask_v.sum()

                # Save interpretable outputs
                os.makedirs(self.args.output_dir, exist_ok=True)
                torch.save({
                    "obs_contrib": obs_contrib,
                    "att_weights": att,
                    "varis": varis,
                    "times": times,
                    "values": values,
                    "var_mean_contrib": var_contrib,
                    "var_counts": counts,
                }, os.path.join(self.args.output_dir, f"{self.args.dataset}_istrats_interpret_infer.pt"))

            # ROC curve
            fpr, tpr, _ = roc_curve(true, pred)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Sepsis Classification')
            plt.legend()
            plt.savefig(os.path.join(self.args.output_dir, f"{self.args.dataset}_roc.png"))
            plt.close()

            # PR curve
            plt.figure()
            plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve for Sepsis Classification')
            plt.legend()
            plt.savefig(os.path.join(self.args.output_dir, f"{self.args.dataset}_pr.png"))
            plt.close()

            # Prediction histograms
            plt.figure(figsize=(8, 5))
            plt.hist(pred[true == 0], bins=50, alpha=0.6, label='Negative', density=True)
            plt.hist(pred[true == 1], bins=50, alpha=0.6, label='Positive', density=True)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.title('Prediction Distribution')
            plt.legend()
            plt.savefig(os.path.join(self.args.output_dir, f"{self.args.dataset}_hist.png"))
            plt.close()

            # Reliability diagram
            prob_true, prob_pred = calibration_curve(true, pred, n_bins=10)
            plt.figure()
            plt.plot(prob_pred, prob_true, marker='o', label='Model')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
            plt.xlabel('Mean predicted probability')
            plt.ylabel('Fraction of positives')
            plt.title('Reliability Diagram')
            plt.legend()
            plt.savefig(os.path.join(self.args.output_dir, f"{self.args.dataset}_calibration.png"))
            plt.close()

        return result
