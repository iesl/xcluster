"""
Copyright (C) 2017 University of Massachusetts Amherst.
This file is part of "xcluster"
http://github.com/iesl/xcluster
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import argparse
import time, datetime
import os
import sys
import errno

from xcluster.models.PNode import PNode

from xcluster.utils.serialize_trees import \
    serliaze_tree_to_file_with_point_ids,\
    serliaze_collapsed_tree_to_file_with_point_ids
from xcluster.models.pruning_heuristics import pick_k_min_dist, \
    pick_k_max_dist, \
    pick_k_point_counter, pick_k_local_mean_cost, pick_k_global_k_mean_cost, \
    pick_k_approx_km_cost


def mkdir_p_safe(dir):
    try:
        os.makedirs(dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def load_data(filename):
    with open(filename, 'r') as f:
        for line in f:
            splits = line.strip().split('\t')
            pid, l, vec = splits[0], splits[1], np.array([float(x)
                                                          for x in splits[2:]])
            yield ((vec, l, pid))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PERCH clustering.')
    parser.add_argument('--input', '-i', type=str,
                        help='Path to the dataset.')
    parser.add_argument('--outdir', '-o', type=str,
                        help='the output directory')
    parser.add_argument('--algorithm', '-a', type=str,
                        help='The name of the algorithm to evaluate.')
    parser.add_argument('--dataset', '-n', type=str,
                        help='The name of the dataset.')
    parser.add_argument('--max_leaves', '-L', type=str,
                        help='The maximum number of leaves.', default=None)
    parser.add_argument('--clusters', '-k', type=str,
                        help='The number of clusters to pick.', default=None)
    parser.add_argument('--pick_k', '-m', type=str,
                        help='The heuristic by which to pick clusters',
                        default=None)
    parser.add_argument('--exact_dist_thres', '-e', type=int,
                        help='# of points to search using exact dist threshold',
                        default=10)

    args = parser.parse_args()

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H:%M:%S')
    exp_dir_base = os.path.join(args.outdir)
    mkdir_p_safe(exp_dir_base)

    clustering_time_start = time.time()
    clustering_time_per_point = []
    counter = 0
    L = int(args.max_leaves) if args.max_leaves and \
                                args.max_leaves.lower() != "none" \
        else float("Inf")
    L_was_defined = True if args.max_leaves and \
                            args.max_leaves.lower() != "none" else False
    collapsibles = [] if L < float("Inf") else None
    exact_dist_thresh = args.exact_dist_thres
    root = PNode(exact_dist_thres=10)
    for pt in load_data(args.input):
        pt_start = time.time()
        root = root.insert(pt, collapsibles=collapsibles, L=L)
        pt_end = time.time()
        clustering_time_per_point.append(pt_end - pt_start)
        if counter % 100 == 0:
            sys.stdout.write("Points %d || Clustering Time: %f || Avg per point"
                             " %f\n" % (
                counter, time.time() - clustering_time_start, sum(
                    clustering_time_per_point) / len(
                    clustering_time_per_point)))
            sys.stdout.flush()
        counter += 1

    clustering_time_end = time.time()
    clustering_time_elapsed = clustering_time_end - clustering_time_start
    sys.stdout.write("Clustering time %s\t%s\t%f\n" % (
        args.algorithm, args.dataset, clustering_time_elapsed))
    sys.stdout.flush()

    # First save the tree structure to a file to evaluate dendrogram purity.
    if L_was_defined:
        serliaze_collapsed_tree_to_file_with_point_ids(
            root, os.path.join(exp_dir_base, 'tree.tsv'))
    else:
        serliaze_tree_to_file_with_point_ids(
            root, os.path.join(exp_dir_base, 'tree.tsv'))

    pick_k_time = 0
    K = int(args.clusters) if args.clusters and \
                              args.clusters.lower() != "none" else None
    if K:
        start_pick_k = time.time()
        if collapsibles is None:
            collapsibles = root.find_collapsibles()
        if not L_was_defined:
            L = root.point_counter

        pick_k_method = args.pick_k if args.pick_k else 'approxKM'
        if pick_k_method == 'approxKM':
            pick_k_approx_km_cost(root, collapsibles, L, K)
        elif pick_k_method == 'pointCounter':
            pick_k_point_counter(root, collapsibles, K)
        elif pick_k_method == 'globalKM':
            pick_k_global_k_mean_cost(root, collapsibles, L, K)
        elif pick_k_method == 'localKM':
            pick_k_local_mean_cost(root, collapsibles, L, K)
        elif pick_k_method == 'maxD':
            pick_k_max_dist(root, collapsibles, L, K)
        elif pick_k_method == 'minD':
            pick_k_min_dist(root, collapsibles, L, K)
        else:
            print('UNKNOWN PICK K METHOD USING approxKM')
            pick_k_approx_km_cost(root, collapsibles, L, K)

        end_pick_k = time.time()
        pick_k_time = end_pick_k-start_pick_k

    # Record running times
    with open(os.path.join(exp_dir_base, 'running_time.tsv'), 'w') as fout:
        fout.write('%s\t%s\t%f\n' % (
            args.algorithm, args.dataset, clustering_time_elapsed))
        fout.write('%s-pick-k\t%s\t%f\n' % (
            args.algorithm, args.dataset,
            clustering_time_elapsed + pick_k_time))

    # Write clustering
    clustering = root.clusters()
    predicted_clustering = []
    gold_clustering = []
    idx = 0
    c_idx = 0
    for c in clustering:
        pts = [pt for l in c.leaves() for pt in l.pts]
        for pt in pts:
            predicted_clustering.append((idx, c_idx))
            gold_clustering.append((idx, pt[1]))
            idx += 1
        c_idx += 1

    with open(os.path.join(exp_dir_base, 'predicted.txt'), 'w') as fout:
        for p in predicted_clustering:
            fout.write('{}\t{}\n'.format(p[0], p[1]))
    with open(os.path.join(exp_dir_base, 'gold.txt'), 'w') as fout:
        for g in gold_clustering:
            fout.write('{}\t{}\n'.format(g[0], g[1]))
    serliaze_collapsed_tree_to_file_with_point_ids(
        root, os.path.join(exp_dir_base, 'tree-pick-k.tsv'))
