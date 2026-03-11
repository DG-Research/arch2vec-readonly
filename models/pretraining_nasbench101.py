import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from models.model import Model, VAEReconstructed_Loss
from utils.utils import load_json, save_checkpoint_vae, preprocessing
from utils.utils import get_val_acc_vae
from models.configs import configs
import argparse
from nasbench import api
from nasbench.lib import graph_util
import itertools
import csv
from pathlib import Path
import random


RESULTS_HEADERS = [
    "seed",
    "validity",
    "uniqueness_bucket",
    "uniqueness_str",
    "absolute_uniqueness_str",
    "novelty_bucket",
    "novelty_str",
    "absolute_novelty_str",
    "validity_counter",
    "uniqueness_bucket_counter",
    "uniqueness_str_counter",
    "novelty_counter",
    "novelty_counter_str",
    "non_novelty_counter",
    "non_novelty_counter_str",
    "actual_novel_bucket",
    "actual_novel_str",
    "max_val",
    "max_test",
    "min_val",
    "min_test",
    "max_val_novel",
    "max_test_novel",
    "min_val_novel",
    "min_test_novel",
    "val_mean",
    "test_mean",
    "novel_val_mean",
    "novel_test_mean",
    "val_median",
    "test_median",
    "val_top_1pct",
    "test_top_1pct",
    "val_top_5pct",
    "test_top_5pct",
    "val_top_10pct",
    "test_top_10pct",
]

def top_pct_mean(values, pct):
    if not values:
        return 0.0
    k = max(1, int(np.ceil(len(values) * (pct / 100.0))))
    top_vals = sorted(values, reverse=True)[:k]
    return float(np.mean(top_vals))

def summarize_accs(values):
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "top_1pct": 0.0,
            "top_5pct": 0.0,
            "top_10pct": 0.0,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "top_1pct": top_pct_mean(values, 1),
        "top_5pct": top_pct_mean(values, 5),
        "top_10pct": top_pct_mean(values, 10),
    }

def transform_operations(max_idx):
    transform_dict =  {0:'input', 1:'conv1x1-bn-relu', 2:'conv3x3-bn-relu', 3:'maxpool3x3', 4:'output'}
    ops = []
    for idx in max_idx:
        ops.append(transform_dict[idx.item()])
    return ops

def _build_dataset(dataset, indices):
    x_adj = []
    x_ops = []
    for ind in indices:
        x_adj.append(torch.Tensor(dataset[str(ind)]['module_adjacency']))
        x_ops.append(torch.Tensor(dataset[str(ind)]['module_operations']))
    x_adj = torch.stack(x_adj)
    x_ops = torch.stack(x_ops)
    return x_adj, x_ops, torch.Tensor(indices)


def sample(dataset, train_split=0.9, seed=0, if_random=False):
    print('random seed: {}'.format(seed))
    if not if_random:
        indices = list(range(len(dataset)))
        train_size = int(len(dataset) * train_split)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        return train_indices, val_indices
    random_shuffle = np.random.RandomState(seed=seed).permutation(range(len(dataset)))
    train_indices = [i for i in random_shuffle[:int(len(dataset) * train_split)]]
    val_indices = [i for i in random_shuffle[int(len(dataset) * train_split):]]
    return train_indices, val_indices


def split_original_dataset_svge(data_path, train_split=0.9, seed=0, if_random=False):
    dataset = load_json(data_path)
    print("original dataset size: {}".format(len(dataset)))
    train_ind_list, val_ind_list = sample(dataset, train_split, seed, if_random)
    x_adj_train, x_ops_train, indices_train = _build_dataset(dataset, train_ind_list)
    x_adj_val, x_ops_val, indices_val = _build_dataset(dataset, val_ind_list)
    return x_adj_train, x_ops_train, indices_train, x_adj_val, x_ops_val, indices_val


def find_train_split(original_data_path, data_seed=0, if_random=False):
    train_bucket = {}
    train_strings = []
    adj, ops, _, _, _, _ = (split_original_dataset_svge(original_data_path, seed=data_seed, if_random=if_random))
    for i in range(len(adj)):
        adj_i = adj[i].int().triu(1).numpy()
        ops_ori = ops[i]
        adj_i, ops_i = np.array(adj_i), ops[i].numpy().tolist()
        ad_str = list(itertools.chain.from_iterable(adj_i))
        ad_str = ''.join([str(int(i)) for i in ad_str])
        op_str = list(itertools.chain.from_iterable(ops_i))
        op_str = ''.join([str(int(i)) for i in op_str])
        string = str(ad_str) + str(op_str)
        train_strings.append(string)
        fingerprint = graph_util.hash_module(adj_i, ops_i)
        if fingerprint not in train_bucket:
            train_bucket[fingerprint] = (adj_i, ops_ori.numpy().astype('int8').tolist())
    return train_bucket, train_strings


def pretraining_model(dataset, cfg, args):
    nasbench = api.NASBench('data/nasbench_only108.tfrecord')
    train_ind_list, val_ind_list = range(int(len(dataset)*0.9)), range(int(len(dataset)*0.9), len(dataset))
    X_adj_train, X_ops_train, indices_train = _build_dataset(dataset, train_ind_list)
    X_adj_val, X_ops_val, indices_val = _build_dataset(dataset, val_ind_list)
    model = Model(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.dim,
                   num_hops=args.hops, num_mlp_layers=args.mlps, dropout=args.dropout, **cfg['GAE']).cpu()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    epochs = args.epochs
    bs = args.bs
    loss_total = []
    for epoch in range(0, epochs):
        chunks = len(train_ind_list) // bs
        if len(train_ind_list) % bs > 0:
            chunks += 1
        X_adj_split = torch.split(X_adj_train, bs, dim=0)
        X_ops_split = torch.split(X_ops_train, bs, dim=0)
        indices_split = torch.split(indices_train, bs, dim=0)
        loss_epoch = []
        Z = []
        for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
            optimizer.zero_grad()
            adj, ops = adj.cpu(), ops.cpu()
            # preprocessing
            adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
            # forward
            ops_recon, adj_recon, mu, logvar = model(ops, adj.to(torch.long))
            Z.append(mu)
            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)
            loss = VAEReconstructed_Loss(**cfg['loss'])((ops_recon, adj_recon), (ops, adj), mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_epoch.append(loss.item())
            if i%1000==0:
                print('epoch {}: batch {} / {}: loss: {:.5f}'.format(epoch, i, chunks, loss.item()))
        Z = torch.cat(Z, dim=0)
        z_mean, z_std = Z.mean(0), Z.std(0)
        validity_counter = 0
        buckets = {}
        model.eval()
        for _ in range(args.latent_points):
            z = torch.randn(7, args.dim).cpu()
            z = z * z_std + z_mean
            op, ad = model.decoder(z.unsqueeze(0))
            op = op.squeeze(0).cpu()
            ad = ad.squeeze(0).cpu()
            max_idx = torch.argmax(op, dim=-1)
            one_hot = torch.zeros_like(op)
            for i in range(one_hot.shape[0]):
                one_hot[i][max_idx[i]] = 1
            op_decode = transform_operations(max_idx)
            ad_decode = (ad>0.5).int().triu(1).numpy()
            ad_decode = np.ndarray.tolist(ad_decode)
            spec = api.ModelSpec(matrix=ad_decode, ops=op_decode)
            # if nasbench.is_valid(spec):
            #     validity_counter += 1
            #     fingerprint = graph_util.hash_module(np.array(ad_decode), one_hot.numpy().tolist())
            #     if fingerprint not in buckets:
            #         buckets[fingerprint] = (ad_decode, one_hot.numpy().astype('int8').tolist())
        validity = validity_counter / args.latent_points
        print('Ratio of valid decodings from the prior: {:.4f}'.format(validity))
        print('Ratio of unique decodings from the prior: {:.4f}'.format(len(buckets) / (validity_counter+1e-8)))
        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = get_val_acc_vae(model, cfg, X_adj_val, X_ops_val, indices_val)
        print('validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val))
        print('epoch {}: average loss {:.5f}'.format(epoch, sum(loss_epoch)/len(loss_epoch)))
        loss_total.append(sum(loss_epoch) / len(loss_epoch))
        save_checkpoint_vae(model, optimizer, epoch, sum(loss_epoch) / len(loss_epoch), args.dim, args.name, args.dropout, args.seed)
    print('loss for epochs: \n', loss_total)
    get_results_inference(model, z_mean, z_std, nasbench)


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def append_row_to_csv(file_path, headers, row):
    p = Path(file_path)
    write_header = not p.exists() or p.stat().st_size == 0
    # Ensure parent directory exists
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(row)

def get_results_inference(model, z_mean, z_std, nasbench):
    validity_counter = 0
    non_novelty_counter = 0
    novelty_counter = 0
    non_novelty_counter_str = 0
    novelty_counter_str = 0
    valid_graphs = []
    novel_bucket_list = []
    novel_str = []
    buckets = {}

    val_accs = []
    test_accs = []
    novel_val_accs = []
    novel_test_accs = []

    max_val = 0.0
    max_test = 0.0
    max_val_novel = 0.0
    max_test_novel = 0.0
    min_val = 100.0
    min_test = 100.0
    min_val_novel = 100.0
    min_test_novel = 100.0


    train_bucket, train_strings = find_train_split(args.data, data_seed=args.seed, if_random=True)
    model.eval()
    for _ in range(args.latent_points):
        z = torch.randn(7, args.dim).cpu()
        z = z * z_std + z_mean
        op, ad = model.decoder(z.unsqueeze(0))
        op = op.squeeze(0).cpu()
        ad = ad.squeeze(0).cpu()
        max_idx = torch.argmax(op, dim=-1)
        one_hot = torch.zeros_like(op)
        for i in range(one_hot.shape[0]):
            one_hot[i][max_idx[i]] = 1
        op_decode = transform_operations(max_idx)
        ad_decode = (ad > 0.5).int().triu(1).numpy()
        ad_decode = np.ndarray.tolist(ad_decode)
        spec = api.ModelSpec(matrix=ad_decode, ops=op_decode)
        if nasbench.is_valid(spec):
            validity_counter += 1
            fingerprint = graph_util.hash_module(np.array(ad_decode), one_hot.numpy().tolist())
            if fingerprint not in buckets:
                buckets[fingerprint] = (ad_decode, one_hot.numpy().astype('int8').tolist())
            if fingerprint not in train_bucket:
                novelty_counter += 1
                novel_bucket_list.append(fingerprint)
            if fingerprint in train_bucket:
                non_novelty_counter += 1

            ad_str = list(itertools.chain.from_iterable(ad_decode))
            ad_str = ''.join([str(int(i)) for i in ad_str])
            op_str = list(itertools.chain.from_iterable(one_hot))
            op_str = ''.join([str(int(i)) for i in op_str])
            string = str(ad_str) + str(op_str)
            valid_graphs.append(string)

            data_query = nasbench.query(spec)
            val_acc = round(float(str(data_query["validation_accuracy"])) * 100, 2)
            test_acc = round(float(str(data_query["test_accuracy"])) * 100, 2)
            val_accs.append(val_acc)
            test_accs.append(test_acc)
            if val_acc > max_val:
                max_val = val_acc
                max_test = test_acc
            if val_acc < min_val:
                min_val = val_acc
                min_test = test_acc
            if string in train_strings:
                non_novelty_counter_str += 1
            if string not in train_strings:
                novelty_counter_str += 1
                novel_val_accs.append(val_acc)
                novel_test_accs.append(test_acc)

                if val_acc > max_val_novel:
                    max_val_novel = val_acc
                    max_test_novel = test_acc
                if val_acc < min_val_novel:
                    min_val_novel = val_acc
                    min_test_novel = test_acc

                novel_str.append(string)

    len_X = max(1, args.latent_points)
    val_stats = summarize_accs(val_accs)
    test_stats = summarize_accs(test_accs)
    novel_val_stats = summarize_accs(novel_val_accs)
    novel_test_stats = summarize_accs(novel_test_accs)

    validity = validity_counter / max(1, args.latent_points)
    validity_by_len_x = validity_counter / len_X
    print("validity: ", validity)
    print("validity wrt to len X: ", validity_by_len_x)

    uniqueness_str = len(set(valid_graphs)) / max(1, len(valid_graphs)) if len(valid_graphs) > 0 else 0.0
    uniqueness_str_counter = len(set(valid_graphs))
    print("Uniqueness in terms of strings: ", uniqueness_str)
    uniqueness_bucket = len(buckets) / max(1, validity_counter)
    uniqueness_bucket_counter = len(buckets)
    print('Ratio of unique decodings from the prior: {:.4f}'.format(uniqueness_bucket))
    print("Number of valid graphs generated: ", validity_counter)
    print("Number of graphs in training set: ", non_novelty_counter)
    print("Number of graphs NOT in training set: ", novelty_counter)
    print("Number of strings in training set: ", non_novelty_counter_str)
    print("Number of strings NOT in training set: ", novelty_counter_str)
    novelty_bucket = 1 - (non_novelty_counter / max(1, validity_counter)) if validity_counter > 0 else 0.0
    print("novelty in terms of buckets: ", novelty_bucket)
    novelty_str = 1 - (non_novelty_counter_str / max(1, len(valid_graphs))) if len(valid_graphs) > 0 else 0.0
    print("novelty in terms of strings: ", novelty_str)
    actual_novel_str = len(set(novel_str))
    print("actual valid, unique and novel in terms of strings: ", actual_novel_str)
    actual_novel_bucket = len(set(novel_bucket_list))
    print("actual valid, unique and novel in terms of buckets: ", actual_novel_bucket)

    absolute_uniqueness_str = (validity * uniqueness_str) * 100.0
    print("absolute uniqueness in terms of strings: ", absolute_uniqueness_str)
    absolute_novelty_str = (validity * novelty_str) * 100.0
    print("absolute novelty in terms of strings: ", absolute_novelty_str)
    data = [
        args.seed,
        validity,
        uniqueness_bucket,
        uniqueness_str,
        absolute_uniqueness_str,
        novelty_bucket,
        novelty_str,
        absolute_novelty_str,
        validity_counter,
        uniqueness_bucket_counter,
        uniqueness_str_counter,
        novelty_counter,
        novelty_counter_str,
        non_novelty_counter,
        non_novelty_counter_str,
        actual_novel_bucket,
        actual_novel_str,
        max_val,
        max_test,
        min_val,
        min_test,
        max_val_novel,
        max_test_novel,
        min_val_novel,
        min_test_novel,
        val_stats["mean"],
        test_stats["mean"],
        novel_val_stats["mean"],
        novel_test_stats["mean"],
        val_stats["median"],
        test_stats["median"],
        val_stats["top_1pct"],
        test_stats["top_1pct"],
        val_stats["top_5pct"],
        test_stats["top_5pct"],
        val_stats["top_10pct"],
        test_stats["top_10pct"],
    ]

    append_row_to_csv(args.all_results_file, RESULTS_HEADERS, data)



def get_continuous_ablation_results(latent_path, nasbench):
    validity_counter = 0
    non_novelty_counter = 0
    novelty_counter = 0
    non_novelty_counter_str = 0
    novelty_counter_str = 0
    valid_graphs = []
    novel_bucket_list = []
    novel_str = []
    buckets = {}

    val_accs = []
    test_accs = []
    novel_val_accs = []
    novel_test_accs = []

    max_val = 0.0
    max_test = 0.0
    max_val_novel = 0.0
    max_test_novel = 0.0
    min_val = 100.0
    min_test = 100.0
    min_val_novel = 100.0
    min_test_novel = 100.0


    train_bucket, train_strings = find_train_split(args.data, data_seed=args.seed, if_random=True)
    dataset = torch.load(latent_path)

    model = Model(input_dim=5, hidden_dim=128, latent_dim=16, num_hops=5, num_mlp_layers=2, dropout=0,
                       **cfg['GAE']).cpu()
    model.load_state_dict(torch.load("pretrained/dim-16/model-nasbench-101.pt")['model_state'])
    model.eval()
    with torch.no_grad():

        for i in dataset:
            z = i.unsqueeze(0)
            print(z.shape)
            op, ad = model.decoder(z)
            op = op.squeeze(0).cpu()
            ad = ad.squeeze(0).cpu()
            max_idx = torch.argmax(op, dim=-1)
            one_hot = torch.zeros_like(op)
            for i in range(one_hot.shape[0]):
                one_hot[i][max_idx[i]] = 1
            op_decode = transform_operations(max_idx)
            ad_decode = (ad > 0.5).int().triu(1).numpy()
            ad_decode = np.ndarray.tolist(ad_decode)
            spec = api.ModelSpec(matrix=ad_decode, ops=op_decode)
            if nasbench.is_valid(spec):
                validity_counter += 1
                fingerprint = graph_util.hash_module(np.array(ad_decode), one_hot.numpy().tolist())
                if fingerprint not in buckets:
                    buckets[fingerprint] = (ad_decode, one_hot.numpy().astype('int8').tolist())
                if fingerprint not in train_bucket:
                    novelty_counter += 1
                    novel_bucket_list.append(fingerprint)
                if fingerprint in train_bucket:
                    non_novelty_counter += 1

                ad_str = list(itertools.chain.from_iterable(ad_decode))
                ad_str = ''.join([str(int(i)) for i in ad_str])
                op_str = list(itertools.chain.from_iterable(one_hot))
                op_str = ''.join([str(int(i)) for i in op_str])
                string = str(ad_str) + str(op_str)
                valid_graphs.append(string)

                data_query = nasbench.query(spec)
                val_acc = round(float(str(data_query["validation_accuracy"])) * 100, 2)
                test_acc = round(float(str(data_query["test_accuracy"])) * 100, 2)
                val_accs.append(val_acc)
                test_accs.append(test_acc)
                if val_acc > max_val:
                    max_val = val_acc
                    max_test = test_acc
                if val_acc < min_val:
                    min_val = val_acc
                    min_test = test_acc
                if string in train_strings:
                    non_novelty_counter_str += 1
                if string not in train_strings:
                    novelty_counter_str += 1
                    novel_val_accs.append(val_acc)
                    novel_test_accs.append(test_acc)

                    if val_acc > max_val_novel:
                        max_val_novel = val_acc
                        max_test_novel = test_acc
                    if val_acc < min_val_novel:
                        min_val_novel = val_acc
                        min_test_novel = test_acc

                    novel_str.append(string)

    len_X = max(1, args.latent_points)
    val_stats = summarize_accs(val_accs)
    test_stats = summarize_accs(test_accs)
    novel_val_stats = summarize_accs(novel_val_accs)
    novel_test_stats = summarize_accs(novel_test_accs)

    validity = validity_counter / max(1, args.latent_points)
    validity_by_len_x = validity_counter / len_X
    print("validity: ", validity)
    print("validity wrt to len X: ", validity_by_len_x)

    uniqueness_str = len(set(valid_graphs)) / max(1, len(valid_graphs)) if len(valid_graphs) > 0 else 0.0
    uniqueness_str_counter = len(set(valid_graphs))
    print("Uniqueness in terms of strings: ", uniqueness_str)
    uniqueness_bucket = len(buckets) / max(1, validity_counter)
    uniqueness_bucket_counter = len(buckets)
    print('Ratio of unique decodings from the prior: {:.4f}'.format(uniqueness_bucket))
    print("Number of valid graphs generated: ", validity_counter)
    print("Number of graphs in training set: ", non_novelty_counter)
    print("Number of graphs NOT in training set: ", novelty_counter)
    print("Number of strings in training set: ", non_novelty_counter_str)
    print("Number of strings NOT in training set: ", novelty_counter_str)
    novelty_bucket = 1 - (non_novelty_counter / max(1, validity_counter)) if validity_counter > 0 else 0.0
    print("novelty in terms of buckets: ", novelty_bucket)
    novelty_str = 1 - (non_novelty_counter_str / max(1, len(valid_graphs))) if len(valid_graphs) > 0 else 0.0
    print("novelty in terms of strings: ", novelty_str)
    actual_novel_str = len(set(novel_str))
    print("actual valid, unique and novel in terms of strings: ", actual_novel_str)
    actual_novel_bucket = len(set(novel_bucket_list))
    print("actual valid, unique and novel in terms of buckets: ", actual_novel_bucket)

    absolute_uniqueness_str = (validity * uniqueness_str) * 100.0
    print("absolute uniqueness in terms of strings: ", absolute_uniqueness_str)
    absolute_novelty_str = (validity * novelty_str) * 100.0
    print("absolute novelty in terms of strings: ", absolute_novelty_str)
    data = [
        args.seed,
        validity,
        uniqueness_bucket,
        uniqueness_str,
        absolute_uniqueness_str,
        novelty_bucket,
        novelty_str,
        absolute_novelty_str,
        validity_counter,
        uniqueness_bucket_counter,
        uniqueness_str_counter,
        novelty_counter,
        novelty_counter_str,
        non_novelty_counter,
        non_novelty_counter_str,
        actual_novel_bucket,
        actual_novel_str,
        max_val,
        max_test,
        min_val,
        min_test,
        max_val_novel,
        max_test_novel,
        min_val_novel,
        min_test_novel,
        val_stats["mean"],
        test_stats["mean"],
        novel_val_stats["mean"],
        novel_test_stats["mean"],
        val_stats["median"],
        test_stats["median"],
        val_stats["top_1pct"],
        test_stats["top_1pct"],
        val_stats["top_5pct"],
        test_stats["top_5pct"],
        val_stats["top_10pct"],
        test_stats["top_10pct"],
    ]

    append_row_to_csv(args.all_results_file, RESULTS_HEADERS, data)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument("--seed", type=int, default=4, help="random seed")
    parser.add_argument('--data', type=str, default='data/data.json',
                        help='Data file (default: data.json')
    parser.add_argument('--name', type=str, default='nasbench-101',
                        help='nasbench-101/nasbench-201/darts')
    parser.add_argument('--cfg', type=int, default=4,
                        help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=8,
                        help='training epochs (default: 8)')
    parser.add_argument('--dropout', type=float, default=0, #TODO:CHANGE
                        help='decoder implicit regularization (default: 0.3)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='use input normalization')
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=16,
                        help='feature dimension (default: 16)')
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--latent_points', type=int, default=10000,
                        help='latent points for validaty check (default: 10000)')
    parser.add_argument('--all_results_file', type=str, default='results/nb101_all_results_arch2vec.csv')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    init_seed(args.seed)
    cfg = configs[args.cfg]
    dataset = load_json(args.data)
    print('using {}'.format(args.data))
    print('feat dim {}'.format(args.dim))
    pretraining_model(dataset, cfg, args)

    # nasbench = api.NASBench("data/nasbench_only108.tfrecord")
    #
    # get_continuous_ablation_results("results/continuous_generated_architectures.pt", nasbench)


