import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import numpy as np
from torch import optim
from models.model import Model, VAEReconstructed_Loss
from utils.utils import load_json, save_checkpoint_vae, preprocessing
from utils.utils import get_val_acc_vae, to_ops_nasbench201, is_valid_nasbench201
from models.configs import configs
from nasbench.lib import graph_util
from nasbench import api
import argparse
from pretraining_nasbench101 import summarize_accs, find_train_split, RESULTS_HEADERS, append_row_to_csv
import itertools
import time
import csv


def sync_if_needed():
    if not args.profile or not args.profile_sync:
        return
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif args.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def now():
    sync_if_needed()
    return time.perf_counter()


def percentile(values, p):
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * p))
    return vals[idx]


def append_profile_step(epoch, batch_idx, batch_size, times):
    if not args.profile:
        return
    steps_per_sec = 1.0 / times["step_total_s"] if times["step_total_s"] > 0 else 0.0
    samples_per_sec = batch_size / times["step_total_s"] if times["step_total_s"] > 0 else 0.0
    profile_step_rows.append(
        {
            "row_type": "step",
            "run_id": runtime_run_id,
            "backend": args.device,
            "device": args.device,
            "seed": args.seed,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": global_step_counter,
            "batch_size": batch_size,
            "train_samples": train_samples_count,
            "val_samples": val_samples_count,
            "bs": args.bs,
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "embedding_dim": "",
            "commitment_loss_factor": "",
            "preprocess_s": times["preprocess_s"],
            "forward_s": times["forward_s"],
            "loss_s": times["loss_s"],
            "backward_s": times["backward_s"],
            "optimizer_s": times["optimizer_s"],
            "step_total_s": times["step_total_s"],
            "samples_per_sec": samples_per_sec,
            "steps_per_sec": steps_per_sec,
            "epoch_total_s": "",
            "mean_step_s": "",
            "std_step_s": "",
            "p50_step_s": "",
            "p90_step_s": "",
            "epoch_steps_measured": "",
        }
    )


def append_profile_epoch(epoch, epoch_total_s, step_totals):
    if not args.profile:
        return
    mean_step = sum(step_totals) / len(step_totals) if step_totals else 0.0
    if len(step_totals) > 1:
        variance = sum((x - mean_step) ** 2 for x in step_totals) / (len(step_totals) - 1)
        std_step = variance ** 0.5
    else:
        std_step = 0.0
    steps_per_sec = len(step_totals) / epoch_total_s if epoch_total_s > 0 else 0.0
    samples_per_sec = (len(step_totals) * args.bs) / epoch_total_s if epoch_total_s > 0 else 0.0
    profile_epoch_rows.append(
        {
            "row_type": "epoch_summary",
            "run_id": runtime_run_id,
            "backend": args.device,
            "device": args.device,
            "seed": args.seed,
            "epoch": epoch,
            "batch_idx": "",
            "global_step": "",
            "batch_size": args.bs,
            "train_samples": train_samples_count,
            "val_samples": val_samples_count,
            "bs": args.bs,
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "embedding_dim": "",
            "commitment_loss_factor": "",
            "preprocess_s": "",
            "forward_s": "",
            "loss_s": "",
            "backward_s": "",
            "optimizer_s": "",
            "step_total_s": "",
            "samples_per_sec": samples_per_sec,
            "steps_per_sec": steps_per_sec,
            "epoch_total_s": epoch_total_s,
            "mean_step_s": mean_step,
            "std_step_s": std_step,
            "p50_step_s": percentile(step_totals, 0.50),
            "p90_step_s": percentile(step_totals, 0.90),
            "epoch_steps_measured": len(step_totals),
        }
    )


def write_profile_csv():
    if not args.profile or not args.profile_out:
        return
    out_dir = os.path.dirname(args.profile_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = [
        "row_type",
        "run_id",
        "backend",
        "device",
        "seed",
        "epoch",
        "batch_idx",
        "global_step",
        "batch_size",
        "train_samples",
        "val_samples",
        "bs",
        "epochs",
        "latent_dim",
        "embedding_dim",
        "commitment_loss_factor",
        "preprocess_s",
        "forward_s",
        "loss_s",
        "backward_s",
        "optimizer_s",
        "step_total_s",
        "samples_per_sec",
        "steps_per_sec",
        "epoch_total_s",
        "mean_step_s",
        "std_step_s",
        "p50_step_s",
        "p90_step_s",
        "epoch_steps_measured",
    ]
    with open(args.profile_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in profile_step_rows:
            writer.writerow(row)
        for row in profile_epoch_rows:
            writer.writerow(row)



def _build_dataset(dataset, list):
    indices = np.random.permutation(list)
    X_adj = []
    X_ops = []
    for ind in indices:
        X_adj.append(torch.Tensor(dataset[str(ind)]['module_adjacency']))
        X_ops.append(torch.Tensor(dataset[str(ind)]['module_operations']))
    X_adj = torch.stack(X_adj)
    X_ops = torch.stack(X_ops)
    return X_adj, X_ops, torch.Tensor(indices)


def pretraining_gae(dataset, cfg):
    """
    implementation of model pretraining.
    :param dataset: nas-bench-201
    :param ind_list: a set structure of indices
    :return: the number of samples to achieve global optimum
    """
    global global_step_counter, train_samples_count, val_samples_count

    train_ind_list, val_ind_list = range(int(len(dataset) * 0.9)), range(int(len(dataset) * 0.9), len(dataset))
    X_adj_train, X_ops_train, indices_train = _build_dataset(dataset, train_ind_list)
    X_adj_val, X_ops_val, indices_val = _build_dataset(dataset, val_ind_list)
    train_samples_count = len(indices_train)
    val_samples_count = len(indices_val)

    model = Model(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_hops=args.hops,
        num_mlp_layers=args.mlps,
        dropout=args.dropout,
        **cfg['GAE']
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    epochs = args.epochs
    bs = args.bs
    loss_total = []

    for epoch in range(0, epochs):
        epoch_start = now()
        measured_step_totals = []
        chunks = len(X_adj_train) // bs
        if len(X_adj_train) % bs > 0:
            chunks += 1
        X_adj_split = torch.split(X_adj_train, bs, dim=0)
        X_ops_split = torch.split(X_ops_train, bs, dim=0)
        indices_split = torch.split(indices_train, bs, dim=0)
        loss_epoch = []
        Z = []

        for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
            step_start = now()
            optimizer.zero_grad()
            batch_size = adj.shape[0]
            adj, ops = adj.to(args.device), ops.to(args.device)
            after_to_device = now()

            adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
            after_preprocess = now()

            ops_recon, adj_recon, mu, logvar = model(ops, adj)
            after_forward = now()

            Z.append(mu)
            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)
            loss = VAEReconstructed_Loss(**cfg['loss'])((ops_recon, adj_recon), (ops, adj), mu, logvar)
            after_loss = now()

            loss.backward()
            after_backward = now()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            step_end = now()

            loss_epoch.append(loss.item())
            step_times = {
                "preprocess_s": after_preprocess - after_to_device,
                "forward_s": after_forward - after_preprocess,
                "loss_s": after_loss - after_forward,
                "backward_s": after_backward - after_loss,
                "optimizer_s": step_end - after_backward,
                "step_total_s": step_end - step_start,
            }
            if args.profile and global_step_counter >= args.profile_warmup_steps and (
                (i % max(1, args.profile_log_interval)) == 0
            ):
                measured_step_totals.append(step_times["step_total_s"])
                append_profile_step(epoch, i, batch_size, step_times)
            global_step_counter += 1

            if i % 100 == 0:
                print('epoch {}: batch {} / {}: loss: {:.5f}'.format(epoch, i, chunks, loss.item()))

        epoch_total_s = now() - epoch_start
        append_profile_epoch(epoch, epoch_total_s, measured_step_totals)

        Z = torch.cat(Z, dim=0)
        z_mean, z_std = Z.mean(0), Z.std(0)
        validity_counter = 0
        buckets = {}
        model.eval()
        for _ in range(args.latent_points):
            z = torch.randn(8, args.latent_dim).to(args.device)
            z = z * z_std + z_mean
            op, ad = model.decoder(z.unsqueeze(0))
            op = op.squeeze(0).detach().cpu()
            ad = ad.squeeze(0).detach().cpu()
            max_idx = torch.argmax(op, dim=-1)
            one_hot = torch.zeros_like(op)
            for j in range(one_hot.shape[0]):
                one_hot[j][max_idx[j]] = 1
            op_decode = to_ops_nasbench201(max_idx)
            ad_decode = (ad > 0.5).int().triu(1).numpy()
            ad_decode = np.ndarray.tolist(ad_decode)
            if is_valid_nasbench201(ad_decode, op_decode):
                validity_counter += 1
                fingerprint = graph_util.hash_module(np.array(ad_decode), one_hot.numpy().tolist())
                if fingerprint not in buckets:
                    buckets[fingerprint] = (ad_decode, one_hot.numpy().astype('int8').tolist())
        validity = validity_counter / args.latent_points
        print('Ratio of valid decodings from the prior: {:.4f}'.format(validity))
        print('Ratio of unique decodings from the prior: {:.4f}'.format(len(buckets) / (validity_counter + 1e-8)))

        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = get_val_acc_vae(
            model, cfg, X_adj_val, X_ops_val, indices_val
        )
        print('validation set: acc_ops:{0:.2f}, mean_corr_adj:{1:.2f}, mean_fal_pos_adj:{2:.2f}, acc_adj:{3:.2f}'.format(
            acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val))
        print('epoch {}: average loss {:.5f}'.format(epoch, sum(loss_epoch) / len(loss_epoch)))
        print("reconstructed adj matrix:", adj_recon[1])
        print("original adj matrix:", adj[1])
        print("reconstructed ops matrix:", ops_recon[1])
        print("original ops matrix:", ops[1])
        loss_total.append(sum(loss_epoch) / len(loss_epoch))
        save_checkpoint_vae(model, optimizer, epoch, sum(loss_epoch) / len(loss_epoch), args.latent_dim, args.name, args.dropout, args.seed)

    print('loss for epochs: ', loss_total)
    write_profile_csv()
    get_results_inference(model, z_mean, z_std, nasbench)


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
        z = torch.randn(8, args.latent_dim).cpu()
        z = z * z_std + z_mean
        op, ad = model.decoder(z.unsqueeze(0))
        op = op.squeeze(0).cpu()
        ad = ad.squeeze(0).cpu()
        max_idx = torch.argmax(op, dim=-1)
        one_hot = torch.zeros_like(op)
        for i in range(one_hot.shape[0]):
            one_hot[i][max_idx[i]] = 1
        op_decode = to_ops_nasbench201(max_idx)
        ad_decode = (ad > 0.5).int().triu(1).numpy()
        ad_decode = np.ndarray.tolist(ad_decode)
        if is_valid_nasbench201(ad_decode, op_decode):
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

            if string in train_strings:
                non_novelty_counter_str += 1
            if string not in train_strings:
                novelty_counter_str += 1

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

    model = Model(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                   num_hops=args.hops, num_mlp_layers=args.mlps, dropout=args.dropout, **cfg['GAE']).cpu()
    model.load_state_dict(torch.load("pretrained/dim-16/model-nasbench201.pt")['model_state'])
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
            op_decode = to_ops_nasbench201(max_idx)
            ad_decode = (ad > 0.5).int().triu(1).numpy()
            ad_decode = np.ndarray.tolist(ad_decode)
            if is_valid_nasbench201(ad_decode, op_decode):
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

                if string in train_strings:
                    non_novelty_counter_str += 1
                if string not in train_strings:
                    novelty_counter_str += 1

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
    parser.add_argument('--data', type=str, default='data/cifar10_valid_converged.json')
    parser.add_argument('--cfg', type=int, default=4)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--input_dim', type=int, default=7)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--latent_points', type=int, default=10000)
    parser.add_argument('--name', type=str, default='nasbench201', help='the prefix for the saved check point')
    parser.add_argument('--all_results_file', type=str, default='results/nb201_all_results_arch2vec.csv')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--profile', action='store_true', default=False,
                        help='enable per-step runtime profiling')
    parser.add_argument('--profile_out', type=str, default='',
                        help='path to write profiling CSV')
    parser.add_argument('--profile_warmup_steps', type=int, default=10,
                        help='number of initial steps to skip for profiling')
    parser.add_argument('--profile_log_interval', type=int, default=1,
                        help='log every Nth step after warmup')
    parser.add_argument('--profile_sync', action='store_true', default=True,
                        help='synchronize device around timers for accurate GPU timing')
    parser.add_argument('--no_profile_sync', action='store_false', dest='profile_sync',
                        help='disable synchronization around timers')
    parser.add_argument('--cpu_threads', type=int, default=1,
                        help='number of CPU threads to use when device=cpu')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device == 'cpu' and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, args.cpu_threads))
        except RuntimeError:
            pass

    cfg = configs[args.cfg]
    dataset = load_json(args.data)
    print('using {}'.format(args.data))
    print('feat dim {}'.format(args.latent_dim))

    runtime_run_id = f"{args.name}_seed{args.seed}"
    global_step_counter = 0
    train_samples_count = 0
    val_samples_count = 0
    profile_step_rows = []
    profile_epoch_rows = []

    nasbench = api.NASBench("/data/gpfs/projects/punim1875/arch2vec-readonly/data/nasbench_only108.tfrecord")
    pretraining_gae(dataset, cfg)
