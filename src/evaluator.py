from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np
import matplotlib.pyplot as plt

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
        for start in pbar:
            batch_ind = eval_ind[start:min(num_samples,
                                           start+self.args.eval_batch_size)]
            batch = dataset.get_batch(batch_ind)
            true.append(batch['labels'])
            del batch['labels']
            batch = {k:v.to(self.args.device) for k,v in batch.items()}
            with torch.no_grad():
                pred.append(model(**batch).cpu())
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
            plt.figure()  
            plt.plot(recall, precision, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Breast Cancer Classification')
            plt.legend()
            plt.savefig('../outputs/aumc/strats_final/roc.png')
        
        return result

