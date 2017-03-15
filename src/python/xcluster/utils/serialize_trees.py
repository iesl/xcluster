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

try:
    from Queue import Queue
except:
    pass

try:
    from queue import Queue
except:
    pass

def serliaze_tree_to_file_with_point_ids(root,fn):
    with open(fn,'w') as fout:
        queue = Queue()
        queue.put(root)
        while not queue.empty():
          curr_node = queue.get()
          curr_node_id = curr_node.pts[0][2] if curr_node.is_leaf() else curr_node.id
          fout.write("%s\t%s\t%s\n" % (curr_node_id,curr_node.parent.id if curr_node.parent else "None", curr_node.pts[0][1] if curr_node.is_leaf() else "None"))
          for c in curr_node.children:
              queue.put(c)


def serliaze_tree_to_file(root,fn):
    with open(fn,'w') as fout:
        queue = Queue()
        queue.put(root)
        while not queue.empty():
          curr_node = queue.get()
          fout.write("%s\t%s\t%s\n" % (curr_node.id,curr_node.parent.id if curr_node.parent else "None", curr_node.pts[0][1] if curr_node.is_leaf() else "None"))
          for c in curr_node.children:
              queue.put(c)

def serliaze_collapsed_tree_to_file(root,fn):
    with open(fn,'w') as fout:
        queue = Queue()
        queue.put(root)
        while not queue.empty():
          curr_node = queue.get()
          fout.write("%s\t%s\t%s\n" % (curr_node.id,curr_node.parent.id if curr_node.parent else "None", curr_node.pts[0][1] if curr_node.is_leaf() and not curr_node.is_collapsed else "None"))
          for c in curr_node.children:
              queue.put(c)
          if curr_node.collapsed_leaves is not None:
              for c in curr_node.collapsed_leaves:
                  queue.put(c)

def serliaze_collapsed_tree_to_file_with_point_ids(root,fn):
    with open(fn,'w') as fout:
        queue = Queue()
        queue.put(root)
        while not queue.empty():
          curr_node = queue.get()
          curr_node_id = curr_node.pts[0][2] if curr_node.is_leaf() and not curr_node.is_collapsed else curr_node.id
          fout.write("%s\t%s\t%s\n" % (curr_node_id,curr_node.parent.id if curr_node.parent else "None", curr_node.pts[0][1] if curr_node.is_leaf() and not curr_node.is_collapsed else "None"))
          for c in curr_node.children:
              queue.put(c)
          if curr_node.collapsed_leaves is not None:
              for c in curr_node.collapsed_leaves:
                  queue.put(c)
