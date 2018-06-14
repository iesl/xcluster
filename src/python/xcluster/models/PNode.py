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
from collections import defaultdict
import random
import string
from queue import Queue
from heapq import heappush, heappop
from numba import jit
from bisect import bisect_left

import math


@jit(nopython=True)
def _fast_norm(x):
    """Compute the number of x using numba.

    Args:
    x - a numpy vector (or list).

    Returns:
    The 2-norm of x.
    """
    s = 0.0
    for i in range(len(x)):
        s += x[i] ** 2
    return math.sqrt(s)


@jit(nopython=True)
def _fast_norm_diff(x, y):
    """Compute the norm of x - y using numba.

    Args:
    x - a numpy vector (or list).
    y - a numpy vector (or list).

    Returns:
    The 2-norm of x - y.
    """
    return _fast_norm(x - y)


@jit(nopython=True)
def _fast_min_to_box(mns, mxs, x):
    """Compute the minimum distance of x to a bounding box.

    Take a point x and a bounding box defined by two vectors of the min and max
    coordinate values in each dimension.  Compute the minimum distance of x to
    the box by computing the minimum distance between x and min or max in each
    dimension.  If, for dimension i,

    self.mins[i] <= x[i] <= self.maxes[i],

    then the distance between x and the box in that dimension is 0.

    Args:
    mns - a numpy array of floats representing the minimum coordinate value
        in each dimension of the bounding box.
    mxs - a numpy array of floats representing the maximum coordinate value
        in each dimension of the bounding box.
    x - a numpy array representing the point.

    Returns:
    A float representing the minimum distance betwen x and the box.
    """
    return _fast_norm(np.maximum(np.maximum(x - mxs, mns - x), 0))


@jit(nopython=True)
def _fast_max_to_box(mns, mxs, x):
    """Compute the maximum distance of x to a bounding box.

    Take a point x and a bounding box defined by two vectors of the min and max
    coordinate values in each dimension.  Compute the maximum distance of x to
    the box by computing the maximum distance between x and min or max in each
    dimension.

    Args:
    mns - a numpy array of floats representing the minimum coordinate value
        in each dimension of the bounding box.
    mxs - a numpy array of floats representing the maximum coordinate value
        in each dimension of the bounding box.
    x - a numpy array representing the point.

    Returns:
    A float representing the minimum distance betwen x and the box.
    """
    return _fast_norm(np.maximum(mxs - x, x - mns))


class PNode:
    """PERCH node."""
    def __init__(self, exact_dist_thres=10):
        self.id = "id" + ''.join(random.choice(
          string.ascii_uppercase + string.digits) for _ in range(12))
        self.children = []
        self.parent = None
        self.num = -1  # The order of this node in the tree.
        self.maxes = None
        self.mins = None
        self.pts = []  # each pt # is a tuple of (pt, label).
        self.children_min_d = 0.0
        self.children_max_d = 0.0
        self.point_counter = 0
        self.collapsed_leaves = None
        self.is_collapsed = False
        self.deleted = False
        # With fewer than this many pts compute the min/max distances exactly.
        self.exact_dist_threshold = exact_dist_thres

    def __lt__(self, other):
        """An arbitrary way to determine an order when comparing 2 nodes."""
        return self.id < other.id

    def insert(self, pt, collapsibles=None, L=float("Inf")):
        """Insert a new pt into the tree.

        Apply recurse masking and balance rotations where appropriate.

        Args:
        pt - a tuple of numpy array, class label, point id.
        collapsibles - (optional) heap of collapsed nodes.
        L - (optional) maximum number of leaves in the tree.

        Returns:
        A pointer to the root.
        """
        root = self.root()
        if self.pts is not None and len(self.pts) == 0:
            self.add_pt(pt)
            self._update_params_recursively()
            return root
        else:
            curr_node = self.a_star_exact(pt)
            new_leaf = curr_node._split_down(pt)
            ancs = new_leaf.parent._ancestors()
            for a in ancs:
                a.add_pt(pt)
            _ = new_leaf._update_params_recursively()

            if collapsibles is not None:
                heappush(collapsibles, (new_leaf.parent.children_max_d,
                                        new_leaf.parent))
            new_leaf.siblings()[0].recursive_rotate_if_masked(collapsibles)
            new_leaf.siblings()[0].recursive_rotate_if_unbalanced(collapsibles)

            if collapsibles is not None and root.point_counter > L:
                max_d, best = heappop(collapsibles)
                valid = best.valid_collapse()
                up_to_date = max_d == best.children_max_d
                while not valid or not up_to_date:
                    if not up_to_date:
                        heappush(collapsibles, (best.children_max_d, best))
                    max_d, best = heappop(collapsibles)
                    valid = best.valid_collapse()
                    up_to_date = max_d == best.children_max_d
                best.collapse()
                if best.siblings()[0].is_leaf():
                    heappush(collapsibles, (best.parent.children_max_d,
                                            best.parent))
            return new_leaf.root()

    def min_distance(self, x):
        """Compute the minimum distance between a point x and this node.

        Args:
        x - a numpy array of floats.

        Returns:
        A float representing the lower bound.
        """
        if self.pts and self.point_counter == 1:
            return _fast_norm_diff(x, self.pts[0][0])
        elif self.pts:
            mn = float('inf')
            for pt in self.pts:
                d = _fast_norm_diff(x, pt[0])
                if d < mn:
                    mn = d
            return mn
        else:
            return _fast_min_to_box(self.mins, self.maxes, x)

    def max_distance(self, x):
        """Compute the maximum distance between a point x and this node.

        Args:
        x - a tuple whose first entry is a number vector representing the point.

        Returns:
        A float representing the upper bound.
        """
        if self.pts and self.point_counter == 1:
            return _fast_norm(x - self.pts[0][0])
        elif self.pts:
            mx = 0
            for pt in self.pts:
                d = _fast_norm_diff(x, pt[0])
                if d > mx:
                    mx = d
            return mx
        else:
            return _fast_max_to_box(self.mins, self.maxes, x)

    def _update(self):
        """Update self's bounding box and determine if ancestors need update.

        Check if self's children have changed their bounding boxes. If not,
        we're done. If they have changed, update this node's bounding box. Also,
        determine whether this node needs to store points or not (based on the
        exact distance threshold). There are a handful of scenarios here where
        we must re-cache the distance at the parent and grandparent.

        If this node has no children, update its bounding box and its parent's
        cached distances.

        Args:
        None.

        Returns:
        A tuple of this node and a bool that is true if the parent may need an
        update.
        """
        if self.children:
            old_mins = self.mins
            old_maxs = self.maxes
            self.mins = np.min(np.array([child.mins
                                         for child in self.children]), axis=0)
            self.maxes = np.max(np.array([child.maxes
                                          for child in self.children]), axis=0)

            # Since rotations can remove descendants, it is possible to erase
            # self.pts pre-rotation and need to reinstate it post-rotation.
            if not self.pts:
                children_pts = sum([c.point_counter for c in self.children])
                if 0 < children_pts <= self.exact_dist_threshold:
                    self.pts = []
                    for c in self.children:
                        if c.pts:
                            for cc in c.pts:
                                self.pts.append(cc)

            # Update the cached distances at the parent.
            if self.parent:
                self.parent._update_children_min_d()
                self.parent._update_children_max_d()

            # Now check if we need to update the parent (2 condiations):
            # First, find out if self's mins or maxes changed. If they have, we
            # certainly need to update the parent because we need to update its
            # bounding box.
            new_mins_or_maxes = (not np.array_equal(self.mins, old_mins)) or (
                not np.array_equal(self.maxes, old_maxs))

            # Even if self's bounding box didn't change, if self or its sibling
            # have a pts field that is not None, self's parent must update. This
            # is because, the grand parent's cached distance between its
            # children could change.
            # TODO (AK): perhaps an easier condition is to always update the
            # TODO (AK): parent if self.pount_counter is <= than the threshold.
            still_have_pts = self.pts or (
                self.parent and self.siblings()[0].pts)
            return self, new_mins_or_maxes or still_have_pts
        else:
            self.mins = self.pts[0][0]
            self.maxes = self.pts[0][0]
            if self.parent:
                self.parent._update_children_min_d()
                self.parent._update_children_max_d()
            return self, True

    def _update_params_recursively(self):
        """Update a node's parameters recursively.

        Args:
        None - start computation from a node and propagate upwards.

        Returns:
        A pointer to the root.
        """
        _, need_update = self._update()
        curr_node = self
        while curr_node.parent and need_update:
            _, need_update = curr_node.parent._update()
            curr_node = curr_node.parent
        return curr_node

    def _update_children_min_d(self):
        """Update the children_min_d parameter.

        This parameter is a cached computation of the approximate min distance
        between self's children. Find this distance by computing the min dist
        between child1 and child2 and then from child2 to child1 (because it's
        not a symmetric approximation) and taking the max to get the largest
        lower bound.
        """
        if self.children:
            c0 = self.children[0]
            c1 = self.children[1]
            self.children_min_d = max(
                min(c0.min_distance(c1.mins), c0.min_distance(c1.maxes)),
                min(c1.min_distance(c0.mins), c1.min_distance(c0.maxes)))

    def _update_children_max_d(self):
        """Update the children_max_d parameter.

        This parameter is a cached computation of the approximate max distance
        between self.children. I find this distance by computing the max dist
        between child1 and child2 and then from child2 to child1 (because it's
        not a symmetric approximation) and taking the min to get the smallest
        valid lower bound.
        """
        if self.children:
            c0 = self.children[0]
            c1 = self.children[1]
            self.children_max_d = min(
                max(c0.max_distance(c1.mins), c0.max_distance(c1.maxes)),
                max(c1.max_distance(c0.mins), c1.max_distance(c0.maxes)))

    def a_star_exact(self, pt, heuristic=lambda n, x: n.min_distance(x)):
        """A* search for the nearest neighbor of pt in tree rooted at self.

        Args:
        pt - a tuple with the first element a numpy vector of floats.
        heuristic - a function of a node and a point that returns a float.

        Returns:
        A pointer to a node (that contains the nearest neighbor of x).
        """
        dp = pt[0]
        if not self.children:
            return self
        else:
            frontier = []
            priority = heuristic(self, dp)
            heappush(frontier, (priority, self))
            while frontier:
                priority, target = heappop(frontier)
                if target.children:
                    for child in target.children:
                        min_d = heuristic(child, dp)
                        heappush(frontier, (min_d, child))
                else:
                    return target
        assert(False)   # This line should never be executed.

    def a_star_beam(self, pt, heuristic=lambda n, x: n.min_distance(x),
                    beam_width=10):
        """A* search with a maximum beam width specified.

        Preform A* search but only allow a beam_width of nodes to remain in the
        frontier at any time. Specifically, pop all nodes in the frontier, look
        at all of their children and repopulate the frontier with the best
        beam_width of them. Repeat until beam_width leaves are found. Return
        the best one.

        Args:
        pt - a tuple with the first element a numpy vector of floats.
        heuristic - a function of a node and a point that returns a float.
        beam_width - an integer upper bound on the number of explorable nodes.

        Returns:
        The approximate nearest neighbor of pt in the current tree.
        """
        dp = pt[0]
        # Micro-optimization.
        if self.point_counter <= beam_width:
            best_leaf = None
            d = float("inf")
            for l in self.leaves():
                dis = _fast_norm_diff(l.pts[0][0], dp)
                if dis < d:
                    d = dis
                    best_leaf = l
            return best_leaf

        best_leaves_so_far = []
        if not self.children:
            return self
        else:
            priorities = []
            frontier = []
            priority = heuristic(self, dp)

            ind = bisect_left(priorities, priority)
            frontier.insert(ind, self)
            priorities.insert(ind, priority)
            while len(best_leaves_so_far) < beam_width and frontier:
                to_explore = []
                while frontier:
                    to_explore.append(frontier.pop(0))
                    _ = priorities.pop(0)
                while to_explore:
                    target = to_explore.pop(0)
                    if target.children:
                        for child in target.children:
                            min_d = heuristic(child, dp)
                            ind = bisect_left(priorities, min_d)
                            frontier.insert(ind, child)
                            priorities.insert(ind, min_d)
                    else:
                        heappush(best_leaves_so_far,
                                 (_fast_norm_diff(target.pts[0][0], dp),
                                  target))
                while len(frontier) > beam_width:
                    frontier.pop()
                    priorities.pop()
            return heappop(best_leaves_so_far)[1]
        assert(False) # You can never reach this line.

    def add_child(self, new_child):
        """Add a PNode as a child of this node (i.e., self).

        Args:
        new_child - a PNode.

        Returns:
        A pointer to self with modifications to self and new_child.
        """
        new_child.parent = self
        self.children.append(new_child)
        return self

    def add_pt(self, pt):
        """Add a data point to this node.

        Increment the point counter. If the number of points at self is less
        than or equal to the exact distance threshold, add pt to self.pts.
        Otherwise, set self.pts to be None.

        Args:
        pt - the data point we are adding.

        Returns:
        A point to this node (i.e., self). Self now "contains" pt.
        """
        self.point_counter += 1.0
        if self.pts is not None and \
                        self.point_counter > self.exact_dist_threshold:
            self.pts = None
        elif self.pts is not None:
            self.pts.append(pt)
        return self

    def _split_down(self, pt):
        """
        Create a new node for pt and a new parent with self and pt as children.

        Create a new node housing pt. Then create a new internal node. Add the
        node housing pt as a child of the new internal node. Then, disconnect
        self from its parent and make it a child of the new internal node.
        Finally, make the new internal node a child of self's old parent. Note:
        while this modifies the tree, nodes are NOT UPDATED in this procedure.

        Args:
        pt - the pt to be added.

        Returns:
        A pointer to the new node containing pt.
        """
        new_internal = PNode(exact_dist_thres=self.exact_dist_threshold)
        # If we are splitting down from a collapsed node, then the pts array
        # is already set to None and we shouldn't instantiate one.
        if self.pts is not None:
            new_internal.pts = self.pts[:]  # Copy points to the new node.
        else:
            new_internal.pts = None
        new_internal.point_counter = self.point_counter

        if self.parent:
            self.parent.add_child(new_internal)
            self.parent.children.remove(self)
            new_internal.add_child(self)
        else:
            new_internal.add_child(self)

        new_leaf = PNode(exact_dist_thres=self.exact_dist_threshold)
        new_leaf.add_pt(pt)  # This updates the points counter.
        new_internal.add_child(new_leaf)
        new_internal.add_pt(pt) # This updates the points counter.
        return new_leaf

    def _rotate(self):
        """Rotate self.

        This essentially swaps the position of self's sibling and self's aunt
        in the tree.

        Args:
        None

        Returns:
        None
        """
        aunt = self.aunts()[0]
        sibling = self.siblings()[0]
        grand_parent = self.parent.parent
        self.parent.deleted = True

        # Make the aunt and self have the same parent
        new_parent = PNode(exact_dist_thres=self.exact_dist_threshold)
        new_parent.pts = None
        new_parent.add_child(aunt)
        new_parent.add_child(self)
        new_parent.point_counter = aunt.point_counter + self.point_counter

        # Set the children of the grandparent to be the new_parent and
        # self's sibling.
        grand_parent.children = []
        grand_parent.add_child(new_parent)
        grand_parent.add_child(sibling)

        # Update cached distances. Other cached distances will be updated later.
        self.parent._update_children_min_d()
        self.parent._update_children_max_d()
        new_parent._update()

    def recursive_rotate_if_masked(self, collapsibles=None):
        """Rotate recursively if masking detected.

        Check if the current node is masked and if so rotate. Recursively apply
        this check to each node in the tree with early stopping if no masking
        is detected.

        Args:
        collapsibles - (optional) a heap; only specified if running in collapsed
                     mode.

        Returns:
        None.
        """
        curr_node = self
        masked = True  # Init value.
        r = curr_node.root()
        while curr_node != r and masked:
            if curr_node.parent and curr_node.parent.parent and \
                    curr_node.is_closer_to_aunt():
                curr_node._rotate()
                # Collapsed mode.
                if collapsibles is not None and curr_node.is_leaf() and \
                        curr_node.siblings()[0].is_leaf():
                    heappush(collapsibles, (curr_node.parent.children_max_d,
                                            curr_node.parent))
                curr_node = curr_node.parent
            elif curr_node.children:
                masked = False  # This enables early stopping.
                curr_node = curr_node.parent
            else:
                curr_node = curr_node.parent

    def recursive_rotate_if_unbalanced(self, collapsibles=None):
        """Rotate recursively if balance candidate detected.

        Check if rotating the current node would increase balance in the tree if
        rotated but would also not enduce any additional masking. If so, rotate.
        Apply the check recursively up the tree.

        Args:
        collapsibles - (optional) a heap; only specified if running in collapsed
                     mode.

        Returns:
        None.
        """
        curr_node = self
        r = curr_node.root()
        while curr_node != r:
            sibling = curr_node.siblings()[0]
            rotate_order = sorted([sibling, curr_node],
                                  key=lambda x: x.point_counter)
            if curr_node.parent and curr_node.parent.parent and \
                    rotate_order[0].can_rotate_for_balance():
                rotate_order[0]._rotate()
                if collapsibles is not None and rotate_order[0].is_leaf() and \
                        rotate_order[0].siblings()[0].is_leaf():
                    heappush(collapsibles,
                             (rotate_order[0].parent.children_max_d,
                              rotate_order[0].parent))
                curr_node = rotate_order[0].parent
            elif curr_node.parent and curr_node.parent.parent and \
                    rotate_order[1].can_rotate_for_balance():
                rotate_order[1]._rotate()
                if collapsibles is not None and rotate_order[1].is_leaf() and \
                        rotate_order[1].siblings()[0].is_leaf():
                    heappush(collapsibles,
                             (rotate_order[1].parent.children_max_d,
                              rotate_order[1].parent))
                curr_node = rotate_order[1].parent
            else:
                curr_node = curr_node.parent

    def purity(self, cluster=None):
        """Compute the purity of this node.

        To compute purity, count the number of points in this node of each
        cluster label. Find the label with the most number of points and divide
        bythe total number of points under this node.

        Args:
        cluster - (optional) str, compute purity with respect to this cluster.

        Returns:
        A float representing the purity of this node.
        """
        if cluster:
            pts = [p for l in self.leaves() for p in l.pts]
            return float(len([pt for pt in pts
                              if pt[1] == cluster])) / len(pts)
        else:
            label_to_count = self.class_counts()
        return max(label_to_count.values()) / sum(label_to_count.values())

    def clusters(self):
        return self.true_leaves()

    def class_counts(self):
        """Produce a map from label to the # of descendant points with label."""
        label_to_count = defaultdict(float)
        pts = [p for l in self.leaves() for p in l.pts]
        for x in pts:
            if len(x) == 3:
                p,l,id = x
            else:
                p,l = x
            label_to_count[l] += 1.0
        return label_to_count

    def pure_class(self):
        """If this node has purity 1.0, return its label; else return None."""
        cc = self.class_counts()
        if len(cc) == 1:
            return list(cc.keys())[0]
        else:
            return None

    def siblings(self):
        """Return a list of my siblings."""
        if self.parent:
            return [child for child in self.parent.children if child != self]
        else:
            return []

    def aunts(self):
        """Return a list of all of my aunts."""
        if self.parent and self.parent.parent:
            return [child for child in self.parent.parent.children
                    if child != self.parent]
        else:
            return []

    def _ancestors(self):
        """Return all of this nodes ancestors in order to the root."""
        anc = []
        curr = self
        while curr.parent:
            anc.append(curr.parent)
            curr = curr.parent
        return anc

    def depth(self):
        """Return the number of ancestors on the root to leaf path."""
        return len(self._ancestors())

    def height(self):
        """Return the height of this node."""
        return max([l.depth() for l in self.leaves()])

    def descendants(self):
        """Return all descendants of the current node."""
        d = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            d.append(n)
            if n.children:
                for c in n.children:
                    queue.put(c)
        return d

    def leaves(self):
        """Return the list of leaves under this node."""
        lvs = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            if n.children:
                for c in n.children:
                    queue.put(c)
            elif n.collapsed_leaves:
                lvs.extend(n.collapsed_leaves)
            else:
                lvs.append(n)
        return lvs

    def true_leaves(self):
        """
        Returns all of the nodes which have no children
        (e.g. data points and collapsed nodes)
        """
        lvs = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            if n.children and not n.is_collapsed:
                for c in n.children:
                    queue.put(c)
            else:
                lvs.append(n)
        return lvs

    def lca(self, other):
        """Compute the lowest common ancestor between this node and other.

        The lowest common ancestor between two nodes is the lowest node
        (furthest distances from the root) that is an ancestor of both nodes.

        Args:
        other - a node in the tree.

        Returns:
        A node in the tree that is the lowest common ancestor between self and
        other.
        """

        ancestors = self._ancestors()
        curr_node = other
        while curr_node not in set(ancestors):
            curr_node = curr_node.parent
        return curr_node

    def root(self):
        """Return the root of the tree."""
        curr_node = self
        while curr_node.parent:
            curr_node = curr_node.parent
        return curr_node

    def is_leaf(self):
        """Returns true if self is a leaf, else false."""
        return len(self.children) == 0

    def is_internal(self):
        """Returns false if self is a leaf, else true."""
        return not self.is_leaf()

    def collapse(self):
        """Collapse self.

        Clear self's children and set self's collapsed leaves to be all of its
        descendant leaves.  Mark it as a collapsed leaf.

        Args:
        None.

        Returns:
        None.
        """
        self.collapsed_leaves = self.leaves()
        for c in self.collapsed_leaves:
            c.parent = self
        self.is_collapsed = True
        self.children = []

    def valid_collapse(self):
        """Returns true if this node can be collapsed.

        To be collapsed:
        1) a node may not be "deleted"
        2) must have two children who are both leaves.
        """
        return not self.deleted and self.children and \
               self.children[0].is_leaf() and self.children[1].is_leaf()

    def is_closer_to_aunt(self):
        """Determine if self is "closer" to its aunt than its sibling.

        Check to see if every point in node is closer to every
        point in its aunt's bounding box than in its siblings bounding box.

        Args:
        None.

        Returns:
        True if self is "closer" to its aunt than its sibling. False, otherwise.
        """
        if self.parent and self.parent.parent:
            aunt = self.aunts()[0]
            aunt_max_dist = max(aunt.max_distance(self.mins),
                                aunt.max_distance(self.maxes))
            other_max_dist = max(self.max_distance(aunt.mins),
                                 self.max_distance(aunt.maxes))
            # Since both max distances are upper bounds, we can safely take the
            # min and it will still be an upper bound.
            smallest_max_dist = min(aunt_max_dist, other_max_dist)

            sibling_min_dist = self.parent.children_min_d
            return smallest_max_dist < sibling_min_dist
        else:
            return False

    def _rotate_without_masking(self):
        """Determine if self can be swapped to improve balance.

        We already know the min distance between node and its sibling. We also
        know that the max distance between self and its sibling is smaller than
        the max distance between self and its aunt (otherwise self would have
        rotated to preserve no masking).  Therefore, we need to check if the max
        distance to self's sibling is smaller than the min distance to the aunt.
        If so, self is unambiguously closer to its sibling and would cause
        masking if rotated; but, if the min distance to the aunt is smaller than
        the max distance to the sibling then we can rotate safely.

        Args:
        node - the node with respect to which we check proximity of bbox.

        Returns:
        True if we can rotate self or false otherwise.
        """
        if self.parent and self.parent.parent:
            aunt = self.aunts()[0]
            sib_max_dist = self.parent.children_max_d
            aunt_min_dist = min(aunt.min_distance(self.mins),
                                aunt.min_distance(self.maxes))
            other_min_dist = min(self.min_distance(aunt.mins),
                                 self.min_distance(aunt.maxes))
            # Since oth min distances are lower bounds, we can use the bigger
            # one safely (it's more accurate).
            largest_aunt_min_dist = max(aunt_min_dist, other_min_dist)
            return sib_max_dist > largest_aunt_min_dist
        else:
            return False

    def _rotate_improves_bal(self):
        """Check if rotating self would produce better balance."""
        if self.parent and self.parent.parent:
            aunt_size = self.aunts()[0].point_counter
            parent_size = self.parent.point_counter
            self_size = self.point_counter
            sibling_size = parent_size - self_size
            new_parent_size = self_size + aunt_size
            bal = min(self_size, sibling_size) / max(self_size, sibling_size) \
                + min(aunt_size, parent_size) / max(aunt_size, parent_size)
            bal_rot = min(self_size, aunt_size) / max(self_size, aunt_size) \
                + min(sibling_size, new_parent_size) / \
                max(sibling_size, new_parent_size)
            return bal < bal_rot
        else:
            return False

    def can_rotate_for_balance(self):
        """Return true if self is a balance candidate.

        Self is a balance candidate if: 1) rotating self improves the balance
        of the tree and 2) rotating self will not mask any other nodes.

        Args:
        None

        Returns:
        True if self can be rotated for balance, false otherwise.
        """
        return self._rotate_improves_bal() and self._rotate_without_masking()

    def find_collapsibles(self):
        """Find all nodes that are collapsible.

        In the case that we run PERCH without keeping track of a collapse
        queue but we'd like to collapse to K clusters anyway, use this function
        to construct the queue.

        Args:
        None.

        Returns:
        A heap of collapsible nodes
        """
        collapsibles = []
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            n = queue.get()
            n_children = n.children
            if n_children:
                if n_children[0].is_leaf() and n_children[1].is_leaf():
                    heappush(collapsibles, (n.children_max_d, n))
                else:
                    for c in n_children:
                        if not c.is_leaf():
                            queue.put(c)
        return collapsibles
