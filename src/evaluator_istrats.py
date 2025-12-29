from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import os

class Evaluator:
    def __init__(self, args):
        self.args = args

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
        true = torch.cat(true)
        pred = torch.cat(pred)

        # Compute metrics
        precision, recall, _ = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true, pred)
        result = {'auroc': roc_auc, 'auprc': pr_auc, 'minrp': minrp}

        if train_step is not None:
            self.args.logger.write(
                'Result on ' + split + ' split at train step '
                + str(train_step) + ': ' + str(result)
            )

        # Inference split: plot ROC/PR and save interpretable outputs
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

            # Plot ROC curve
            fpr, tpr, _ = roc_curve(true.cpu().numpy(), pred.cpu().numpy())
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Sepsis Classification')
            plt.legend()
            plt.savefig(os.path.join(self.args.output_dir, f"{self.args.dataset}_roc.png"))
            plt.close()

            # Plot Precision-Recall curve
            plt.figure()
            plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve for Sepsis Classification')
            plt.legend()
            plt.savefig(os.path.join(self.args.output_dir, f"{self.args.dataset}_pr.png"))
            plt.close()

        return result
