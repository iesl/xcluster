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
from heapq import heappop, heappush


def valid_collapse(node):
    """Returns a boolean indicating if this node is collapseable.

    A node is only collapseable if it:
    1) exists
    2) has children
    3) both of its children are leaves.
    """
    return not node.deleted and node.children and node.children[0].is_leaf() \
           and node.children[1].is_leaf() and node.parent


def pick_k_min_dist(root, collapsables, L, K):
    """Collapse based on children_min_d.

    Reorganize the nodes in collapsibles in order of increasing children_min_d.
    Then, as long as it is valid to collapse the node at the head of the heap,
    collapse it.  Stop collapsing once the number of true leaves in the tree is
    equal to K.

    Args:
    root - the root of the tree.
    collapsibles - a heap of nodes.
    L - the size of the heap
    K - the number of desired clusters at the end of the process.

    Returns:
    The root of the tree after collapsing.
    """
    assert(collapsables is not None)
    assert(K is not None)
    assert(L is not None)

    l = L
    while l > K:
        # compute min distance of new parent,
        # peek at the min dist at top of heap
        # while that node is invalid we want to pop
        # compare the current node min dist to peeked element
        # if min dist of peek is greater, collapse new leaf.parent
        # else pop, collapse it
        # add new_leaf.parent to heap
        min_d, best = heappop(collapsables)
        while not valid_collapse(best):
            min_d, best = heappop(collapsables)
        best.collapse()
        if best.siblings()[0].is_leaf():
            heappush(collapsables, (best.parent.children_min_d,
                                    best.parent))
        l -= 1
    return root


def pick_k_max_dist(root, collapsables, L, K):
    """Collapse based on children_max_d.

    Reorganize the nodes in collapsibles in order of increasing children_max_d.
    Then, as long as it is valid to collapse the node at the head of the heap,
    collapse it.  Stop collapsing once the number of true leaves in the tree is
    equal to K.

    Args:
        root - the root of the tree.
        collapsibles - a heap of nodes.
        L - the size of the heap
        K - the number of desired clusters at the end of the process.

    Returns:
        The root of the tree after collapsing.
    """
    assert(collapsables is not None)
    assert(K is not None)
    assert(L is not None)

    new_heap = []
    while collapsables:
        _, node = heappop(collapsables)
        heappush(new_heap, (node.children_max_d, node))

    l = L
    while l > K:
        max_d, best = heappop(new_heap)
        while not valid_collapse(best):
            min_d, best = heappop(new_heap)
        best.collapse()
        if best.siblings()[0].is_leaf():
            heappush(new_heap, (best.parent.children_max_d,
                                best.parent))
        l -= 1
    return root


def pick_k_point_counter(root, collapsables, K):
    """Collapse based on point_counter.

    Reorganize the nodes in collapsibles in order of increasing point_counter.
    Then, as long as it is valid to collapse the node at the head of the heap,
    collapse it.  Stop collapsing once the number of true leaves in the tree is
    equal to K.

    Args:
    root - the root of the tree.
    collapsibles - a heap of nodes.
    L - the size of the heap
    K - the number of desired clusters at the end of the process.

    Returns:
    The root of the tree after collapsing.
    """
    assert(collapsables is not None)
    assert(K is not None)

    new_heap = []
    while collapsables:
        _, node = heappop(collapsables)
        if valid_collapse(node):
            heappush(new_heap, (node.point_counter, node))

    l = len(root.clusters())
    while l > K:
        max_d, best = heappop(new_heap)
        while not valid_collapse(best):
            _, best = heappop(new_heap)
        best.collapse()
        if best.siblings()[0].is_leaf():
            heappush(new_heap, (best.parent.point_counter,
                                best.parent))
        l -= 1
    return root


def pick_k_local_mean_cost(root, collapsables, L, K):
    """Collapse based on local distance to the mean.

    Reorganize the nodes in collapsibles in order of the sum of distances
    between the node's mean and it's descendant data points.  This is similar to
    the k-means cost associated with the node although it doesn't not consider
    the children's sum of distance costs (and is therefore "local").
    Then, as long as it is valid to collapse the node at the head of the heap,
    collapse it.  Stop collapsing once the number of true leaves in the tree is
    equal to K.

    Args:
    root - the root of the tree.
    collapsibles - a heap of nodes.
    L - the size of the heap
    K - the number of desired clusters at the end of the process.

    Returns:
    The root of the tree after collapsing.
    """
    assert(collapsables is not None)
    assert(K is not None)
    assert(L is not None)

    def local_mean_cost(node):
        my_mean = np.mean([l.pts[0][0] for l in node.leaves()], axis=0)
        cost = np.sum([np.linalg.norm(my_mean - l.pts[0][0])
                       for l in node.leaves()])
        return cost

    new_heap = []
    while collapsables:
        _, node = heappop(collapsables)
        if valid_collapse(node):
            heappush(new_heap, (local_mean_cost(node), node))

    l = L
    while l > K:
        max_d, best = heappop(new_heap)
        while not valid_collapse(best):
            _, best = heappop(new_heap)
        best.collapse()
        if best.siblings()[0].is_leaf():
            heappush(new_heap, (local_mean_cost(best.parent),
                                best.parent))
        l -= 1
    return root


def pick_k_approx_km_cost(root, collapsables, L, K):
    """Collapse based on approx k-means cost.

    Reorganize the nodes in collapsibles in order of max distance between any
    two children (divided by 2) times the point counter. This is a crude
    approximation of the k-means cost at the node.

    Args:
    root - the root of the tree.
    collapsibles - a heap of nodes.
    L - the size of the heap
    K - the number of desired clusters at the end of the process.

    Returns:
    The root of the tree after collapsing.
    """
    assert(collapsables is not None)
    assert(K is not None)
    assert(L is not None)

    def approx_km(node):
        return node.children_max_d * 0.5 * node.point_counter

    new_heap = []
    while collapsables:
        _, node = heappop(collapsables)
        if valid_collapse(node):
            heappush(new_heap, (approx_km(node), node))

    l = L
    while l > K:
        max_d, best = heappop(new_heap)
        while not valid_collapse(best):
            _, best = heappop(new_heap)
        best.collapse()
        if best.siblings()[0].is_leaf():
            heappush(new_heap, (approx_km(best.parent),
                                best.parent))
        l -= 1
    return root


def pick_k_global_k_mean_cost(root, collapsables, L, K):
    """Collapse based on local distance to the mean.

    Reorganize the nodes in collapsibles in order of their k-means cost.
    Then, as long as it is valid to collapse the node at the head of the heap,
    collapse it.  Stop collapsing once the number of true leaves in the tree is
    equal to K.

    Args:
    root - the root of the tree.
    collapsibles - a heap of nodes.
    L - the size of the heap
    K - the number of desired clusters at the end of the process.

    Returns:
    The root of the tree after collapsing.
    """

    assert(collapsables is not None)
    assert(K is not None)
    assert(L is not None)

    def k_mean_cost(node):
        my_mean = np.mean([l.pts[0][0] for l in node.leaves()], axis=0)
        cost = np.sum([np.linalg.norm(my_mean - l.pts[0][0])
                       for l in node.leaves()])
        child1_mean = np.mean([l.pts[0][0] for l in node.children[0].leaves()],
                              axis=0)
        child1_cost = np.sum([np.linalg.norm(child1_mean - l.pts[0][0])
                              for l in node.children[0].leaves()])
        child2_mean = np.mean([l.pts[0][0] for l in node.children[1].leaves()],
                              axis=0)
        child2_cost = np.sum([np.linalg.norm(child2_mean - l.pts[0][0])
                              for l in node.children[1].leaves()])
        return cost - (child1_cost + child2_cost)

    new_heap = []
    while collapsables:
        _, node = heappop(collapsables)
        if valid_collapse(node):
            heappush(new_heap, (k_mean_cost(node), node))

    l = L
    while l > K:
        max_d, best = heappop(new_heap)
        while not valid_collapse(best):
            _, best = heappop(new_heap)
        best.collapse()
        if best.siblings()[0].is_leaf():
            heappush(new_heap, (k_mean_cost(best.parent),
                                best.parent))
        l -= 1
    return root
