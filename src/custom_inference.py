import argparse
import torch
import numpy as np
from tqdm import tqdm

import dataset
from dataset import Dataset
from modeling_strats import Strats
from modeling_gru import GRU_TS
from modeling_tcn import TCN_TS
from modeling_sand import SAND
from modeling_grud import GRUD_TS
from modeling_interpnet import InterpNet
from evaluator import Evaluator
from utils import Logger, set_all_seeds


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset / model
    parser.add_argument('--dataset', type=str, default='aumc')
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--load_ckpt_path', type=str, required=True)
    parser.add_argument('--run', type=str, default='1o10')
    parser.add_argument('--train_frac', type=float, default=0.5)
    parser.add_argument('--model_ckpt_path', type=str, required=True)

    # inference options
    parser.add_argument('--split', type=str, default='infer',
                        choices=['train', 'eval_train', 'val', 'test', 'infer'])
    parser.add_argument('--batch_size', type=int, default=32)

    # strats / istrats
    ##  strats and istrats
    parser.add_argument('--max_obs', type=int, default=880)
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)

    # misc
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_dir_prefix', type=str, default='')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--print_train_loss_every', type=int, default=100)
    parser.add_argument('--validate_after', type=int, default=-1)
    parser.add_argument('--validate_every', type=int, default=None)

    return parser.parse_args()


def load_model(args):
    model_class = {
        'strats': Strats,
        'istrats': Strats,
        'gru': GRU_TS,
        'tcn': TCN_TS,
        'sand': SAND,
        'grud': GRUD_TS,
        'interpnet': InterpNet,
    }

    model = model_class[args.model_type](args)
    state_dict = torch.load(args.model_ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    return model




def main():
    args = parse_args()

    args.logger = Logger(args.output_dir, 'log.txt')
    args.logger.write('\n'+str(args))

    args.device = torch.device(args.device)
    set_all_seeds(args.seed)

    # load dataset
    dataset = Dataset(args)

    # load model
    model = load_model(args)

    # load Evaluator
    evaluator = Evaluator(args)
    
   
    evaluator.evaluate(model, dataset, args.split, train_step=-1)

    




if __name__ == "__main__":
    main()
