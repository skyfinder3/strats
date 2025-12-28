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
        self.args.logger.write('\nEvaluating on split = '+split)
        eval_ind = dataset.splits[split]
        num_samples = len(eval_ind)
        model.eval()

        pbar = tqdm(range(0,num_samples,self.args.eval_batch_size),
                    desc='running forward pass')
        true, pred = [], []

        # for interpretable aggregation
        all_obs_contrib = []
        all_att = []
        all_varis = []
        all_times = []
        all_values = []

        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start+self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            labels = batch['labels']
            true.append(labels)

            if split == 'infer':
                # keep raw tensor for mapping back
                all_varis.append(batch['varis'])
                all_times.append(batch['times'])
                all_values.append(batch['values'])

            del batch['labels']
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            
            with torch.no_grad():
                if split == 'infer':
                    out = model(**batch, return_interpret=True)
                    logits = out['logits'].cpu()
                    pred.append(torch.sigmoid(logits))
                    all_obs_contrib.append(out['obs_contrib'].cpu())
                    all_att.append(out['att_weights'].cpu())
                else:
                    logits_or_loss = model(**batch)
        
                    # in non-infer mode, forward returns loss when labels are given
                    if labels is not None:
                        # recompute probs for metrics
                        # run once more without labels
                        batch_nolab = {**batch}
                        batch_nolab.pop('labels', None)
                        logits = model(**batch_nolab)
                    else:
                        logits = logits_or_loss
                    pred.append(logits.cpu())
        
        true = torch.cat(true)
        pred = torch.cat(pred)

        true, pred = torch.cat(true), torch.cat(pred)
        precision, recall, thresholds = precision_recall_curve(true, pred)
        pr_auc = auc(recall, precision)
        minrp = np.minimum(precision, recall).max()
        roc_auc = roc_auc_score(true, pred)
        result = {'auroc':roc_auc, 'auprc':pr_auc, 'minrp':minrp}
        if train_step is not None:
            self.args.logger.write('Result on '+split+' split at train step '
                              +str(train_step)+': '+str(result))
            
        # Add graphs and visualizations for inference split
        if split == 'infer':
            # Plot the ROC curve
            fpr, tpr, roc_thresholds = roc_curve(true.cpu().numpy(),
                                         pred.cpu().numpy())
            plt.figure()  
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Sepsis Classification')
            plt.legend()
            plt.savefig(self.args.output_dir + f"/{self.args.dataset}_roc.png")
            plt.close()

            plt.figure()
            plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve for Sepsis Classification')
            plt.legend()
            plt.savefig(self.args.output_dir + f"/{self.args.dataset}_pr.png")
            plt.close()

            # aggregate interpretable outputs for istrats
            if self.args.model_type == 'istrats':
                obs_contrib = torch.cat(all_obs_contrib, dim=0)  # (N_total, N_max)
                att = torch.cat(all_att, dim=0)
                varis = torch.cat(all_varis, dim=0)
                times = torch.cat(all_times, dim=0)
                values = torch.cat(all_values, dim=0)

                # example: save per-variable mean contribution
                # mask out padded observations using obs_mask if available
                obs_mask = (values != 0)  # or use batch['obs_mask'] collected similarly
                contrib_flat = obs_contrib[obs_mask]
                varis_flat = varis[obs_mask]

                num_vars = self.args.V
                var_contrib = torch.zeros(num_vars)
                counts = torch.zeros(num_vars)

                for v in range(num_vars):
                    mask_v = (varis_flat == v)
                    if mask_v.any():
                        var_contrib[v] = contrib_flat[mask_v].mean()
                        counts[v] = mask_v.sum()
                
                torch.save({
                    "obs_contrib": obs_contrib,
                    "att_weights": att,
                    "varis": varis,
                    "times": times,
                    "values": values,
                    "var_mean_contrib": var_contrib,
                    "var_counts": counts,
                }, os.path.join(self.args.output_dir,f"{self.args.dataset}_istrats_interpret_infer.pt"))
        
        return result

