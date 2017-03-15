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

from xcluster.models.PNode import PNode
from xcluster.utils.dendrogram_purity import dendrogram_purity
from xcluster.utils.deltasep_utils import create_dataset


def create_trees_w_purity_check(dataset):
    """Create trees over the same points.

    Create n trees, online, over the same dataset. Return pointers to the
    roots of all trees for evaluation.  The trees will be created via the insert
    methods passed in.  After each insertion, verify that the dendrogram purity
    is still 1.0 (perfect).

    Args:
        dataset - a list of points with which to build the tree.

    Returns:
        A list of pointers to the trees constructed via the insert methods
        passed in.
    """
    np.random.shuffle(dataset)
    root = PNode(exact_dist_thres=10)

    for i, pt in enumerate(dataset):
        root = root.insert(pt, collapsibles=None, L=float('inf'))
        if i % 10 == 0:
            assert(dendrogram_purity(root) == 1.0)
            assert(root.point_counter == (i + 1))
    return root


if __name__ == '__main__':
    """Test that PERCH produces perfect trees when data is separable.

    PERCH is guarnateed to produce trees with perfect dendrogram purity if
    thet data being clustered is separable. Here, we generate random separable
    data and run PERCH clustering.  Every 10 points assert that the purity is
    1.0."""
    dimensions = [2, 10, 100, 1000]
    size = 25
    num_clus = 50
    for dim in dimensions:
        print("TESTING DIMENSIONS == %d" % dim)
        dataset = create_dataset(dim, size, num_clusters=num_clus)
        with open('separated-%d.tsv' % dim, 'w') as f:
            for d in dataset:
                f.write('%d\t%d\t%s\n' % (d[2], d[1],
                                          '\t'.join([str(x) for x in d[0]])))
        create_trees_w_purity_check(dataset)
