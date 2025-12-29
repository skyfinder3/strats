import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, brier_score_loss, precision_score
from sklearn.calibration import calibration_curve
import shap

class SHAPModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, values, times, varis, obs_mask, demo):
        # Float inputs require gradients
        values = values.float().requires_grad_(True)
        times = times.float().requires_grad_(True)
        obs_mask = obs_mask.float().requires_grad_(True)
        demo = demo.float().requires_grad_(True)

        # Embedding indices must be Long
        varis = varis.long()

        out = self.model(values, times, varis, obs_mask, demo)

        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


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
        pbar = tqdm(range(0, num_samples, self.args.eval_batch_size), desc='running forward pass')

        true, pred = [], []

        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples, start + self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)

            true.append(batch['labels'])
            del batch['labels']

            batch = {k: v.to(self.args.device) for k, v in batch.items()}

            with torch.no_grad():
                out = model(**batch)
                if out.dim() > 1:
                    out = out.squeeze(-1)
                pred.append(out.cpu())

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

        result = {'auroc': roc_auc, 'auprc': pr_auc, 'minrp': minrp, 'brier': brier, 'ece': ece, 'ppv': ppv}

        if train_step is not None:
            self.args.logger.write(f"Result on {split} split at train step {train_step}: {result}")

        # ======================
        # Inference visualizations
        # ======================
        if split == 'infer':
            # ROC curve
            fpr, tpr, _ = roc_curve(true, pred)
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig(f"{self.args.output_dir}/{self.args.dataset}_roc.png")
            plt.close()

            # PR curve
            plt.figure()
            plt.plot(recall, precision, label=f'PR (AUC={pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.savefig(f"{self.args.output_dir}/{self.args.dataset}_pr.png")
            plt.close()

            # Prediction histograms
            plt.figure(figsize=(8, 5))
            plt.hist(pred[true == 0], bins=50, alpha=0.6, label='Negative', density=True)
            plt.hist(pred[true == 1], bins=50, alpha=0.6, label='Positive', density=True)
            plt.xlabel('Predicted Probability')
            plt.ylabel('Density')
            plt.title('Prediction Distribution')
            plt.legend()
            plt.savefig(f"{self.args.output_dir}/{self.args.dataset}_hist.png")
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
            plt.savefig(f"{self.args.output_dir}/{self.args.dataset}_calibration.png")
            plt.close()

        return result
