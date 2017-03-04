/* Copyright (C) 2017 University of Massachusetts Amherst.
   This file is part of “xcluster”
   http://github.com/iesl/xcluster
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

package xcluster.models

import java.io.{File, PrintWriter}
import java.util.concurrent.ExecutorService
import java.util.{Comparator, PriorityQueue, UUID}

import cc.factorie.util.{JavaHashMap, Threading}
import com.google.common.collect.MinMaxPriorityQueue
import xcluster.utils.{ComparisonCounter, ConsistentId}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * A node in a Perch Cluster tree.
  * This contains all operations for constructing a Perch tree
  * as well as producing a flat clustering.
  *
  * @param dim The dimensionality of the points to be clustered
  * @param exactDistThreshold When a subtree with exactDistThreshold or fewer points, an exact (non-tree search)
  *                           nearest neighbor search is performed.
  */
class PerchNode(dim: Int,exactDistThreshold: Int = 20) extends Comparable[PerchNode] {

  /**
    * Unique id of PerchNode
    */
  val id: UUID = ConsistentId.nextId

  /**
    * The children of a PerchNode
    */
  val children = new ArrayBuffer[PerchNode](2)

  /**
    * The nodes which have been collapsed into this node
    * when running in collapsed mode
    */
  val collapsedLeaves = new ArrayBuffer[PerchNode]()

  // Mutable variables defining the state of the node

  /**
    * Max values of this node's bounding box
    */
  var maxes: Array[Double] = Array.fill(dim)(0.0)
  /**
    * Min values of this node's bounding box
    */
  var mins: Array[Double] = Array.fill(dim)(0.0)

  /**
    * The parent of this node
    */
  var parent: PerchNode = null

  /**
    * A cached computation of the approximate min distance
    *    between this node's children.
    */
  var childrenMinD = 0.0

  /**
    * A cached computation of the approximate max distance
    *    between this node's children.
    */
  var childrenMaxD = 0.0

  /**
    * The number of points stored in all of the descendant leaves
    * of this node.
    */
  var pointCounter = 0

  /**
    * Whether or not this node is collapsed
    */
  var isCollapsed = false

  /**
    * Whether or not this node has been deleted from the tree
    */
  var isDeleted = false

  /**
    * The points sitting at this node.
    */
  var pts = new ArrayBuffer[Point](exactDistThreshold)


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                Distance Approximation Methods                     *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * Compute a lower bound on the minimum distance between the point x
    * and any point in this node by using this node's bounding box.
    * @param x A vector x
    * @return Minimum distance lower bound
    */
  def minDistance(x: Array[Double]): Double = {
    // if there is just one point at this node, compute the distance exactly
    if (this.pts != null && this.pts.length == 1) {
     minusNorm(x,this.pts(0).value)
    // if there are fewer than exactDistThreshold points in this node, compute distance exactly
    } else if (this.pts != null && this.pts.length <= this.exactDistThreshold) {
      var i = 0
      var mn = Double.PositiveInfinity
      while (i < this.pts.length) {
        val d = minusNorm(x,this.pts(i).value)
        if (d < mn) {
          mn = d
        }
        i += 1
      }
      mn
    // otherwise estimate by bounding box (Equation 2)
    } else {
      ComparisonCounter.increment()
      ComparisonCounter.increment()
      var diff = 0.0
      var i = 0
      while (i < this.dim) {
        val d = math.max(math.max(x(i)-this.maxes(i), this.mins(i) - x(i)),0.0)
        diff += d*d
        i += 1
      }
      math.sqrt(diff)
    }
  }

  /**
    * Compute an upper bound on the maximum distance between the point x
    * and any point in this node by using this node's bounding box.
    * @param x A vector x
    * @return Maximum distance upper bound
    */
  def maxDistance(x: Array[Double]): Double = {
    // if there is just one point at this node, compute the distance exactly
    if (this.pts != null && this.pts.length == 1) {
      minusNorm(x, this.pts(0).value)
    // if there are fewer than exactDistThreshold points in this node, compute distance exactly
    } else if (this.pts != null && this.pts.length <= this.exactDistThreshold) {
      var i = 0
      var mx = Double.NegativeInfinity
      while (i < this.pts.length) {
        val d = minusNorm(x, this.pts(i).value)
        if (d > mx) {
          mx = d
        }
        i += 1
      }
      mx
    // otherwise estimate by bounding box (Equation 3)
    } else {
      ComparisonCounter.increment()
      ComparisonCounter.increment()
      var diffs = 0.0
      var i = 0
      while (i < this.dim) {
        val d = math.max(this.maxes(i) - x(i), x(i) - this.mins(i))
        diffs += d * d
        i += 1
      }
      math.sqrt(diffs)
    }
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                   Nearest Neighbor Search Methods                 *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * A* search for the nearest neighbor of pt in tree rooted at self.
    *
    * @param pt Query point
    * @return The nearest neighbor of pt
    */
  def AStar(pt: Point): (PerchNode, Double) = {
    if (this.children.isEmpty) {
      (this, this.minDistance(pt.value))
    } else {
      val frontier = new PriorityQueue[(Double,PerchNode)](PQComparator)
      frontier.add((this.minDistance(pt.value), this))
      while (!frontier.isEmpty) {
        val (priority, target) = frontier.remove()
        if (target.children.nonEmpty) {
          target.children.foreach { child =>
            frontier.add((child.minDistance(pt.value), child))
          }
        } else {
          return (target, priority)
        }
      }
      throw new Exception("Search Error. No leaf found.")
    }
  }

  /**
    * Parallelized A* search for nearest neighbors. Parallelization is
    * achieved by building up a frontier of a certain size and then searching
    * each point in the frontier in parallel, with a final nearest neighbor selected
    * from the independent searches at the end of the procedure.
    *
    * @param pt Query point
    * @param maxFrontierSizeBeforeParallelization The maximum size of the frontier to build before searching in parallel
    *                                             (should be greater than number of threads used) (default 50)
    * @param minPointsForParrallism The minimum number of points in the tree to invoke parallelism (default -1)
    * @param threadpool
    * @return
    */
  def AStarParallel(pt: Point, maxFrontierSizeBeforeParallelization: Int = 50, minPointsForParrallism: Int = -1)(implicit threadpool: ExecutorService): (PerchNode, Double) = {
    if (this.children.isEmpty) {
      (this, this.minDistance(pt.value))
    } else {
      if (this.pointCounter < minPointsForParrallism) {
        this.AStar(pt)
      } else {
        val frontier = new PriorityQueue[(Double,PerchNode)](PQComparator)
        frontier.add((this.minDistance(pt.value), this))
        while (frontier.size < maxFrontierSizeBeforeParallelization) {
          val (priority, target) = frontier.remove()
          if (target.children.nonEmpty) {
            target.children.foreach { child =>
              frontier.add((child.minDistance(pt.value), child))
            }
          } else {
            return (target, priority)
          }
        }
        Threading.parMap(frontier.asScala, threadpool)(priorityAndNode => {
          priorityAndNode._2.AStar(pt)
        }).minBy(_._2)
      }
    }
  }

  /**
    * Approximate nearest neighbor search using beam search.
    *
    * @param pt Query point
    * @param beam Beam width
    * @return approximate nearest neighbor
    */
  def beamSearch(pt: Point, beam: Int = 100): (PerchNode,Double) = {
    if (this.children.isEmpty) {
      (this, this.minDistance(pt.value))
    } else {
      var frontier = MinMaxPriorityQueue.orderedBy(PQComparator).maximumSize(beam).create[(Double,PerchNode)]()
      frontier.add((this.minDistance(pt.value), this))
      val bestLeavesSoFar = MinMaxPriorityQueue.orderedBy(PQComparator).maximumSize(beam).create[(Double,PerchNode)]()
      while ((bestLeavesSoFar.size() < beam) && !frontier.isEmpty) {
        val newFrontier = MinMaxPriorityQueue.orderedBy(PQComparator).maximumSize(beam).create[(Double,PerchNode)]()
        // explore in a breadth first approach
        val toExplore = frontier
        toExplore.asScala.foreach{
          case (priority,target) =>
            if (target.children.nonEmpty) {
              target.children.foreach { child =>
                newFrontier.add((child.minDistance(pt.value), child))
              }
            } else {
              bestLeavesSoFar.add((priority, target))
            }
        }
        frontier = newFrontier
      }
      val res = bestLeavesSoFar.poll()
      (res._2,res._1)
    }
  }

  /**
    * Approximate nearest neighbor search using a parallelized beam search.
    *
    * @param pt Query point
    * @param beamSize Beam width per thread. Multiply this value by numThreads for comparable value to single threaded version.
    * @return approximate nearest neighbor of pt
    */
  def beamSearchParallel(pt: Point, beamSize: Int = 4, numThreads: Int = 24, maxFrontierSizeBeforeParallelization: Int = 8, minPointsForParrallism: Int = -1)(implicit threadpool: ExecutorService): (PerchNode, Double) = {
    if (this.children.isEmpty) {
      (this, this.minDistance(pt.value))
    } else {
      if (this.pointCounter < minPointsForParrallism) {
        this.AStar(pt)
      } else {
        val frontier = new PriorityQueue[(Double,PerchNode)](PQComparator)
        frontier.add((this.minDistance(pt.value), this))
        while (frontier.size < maxFrontierSizeBeforeParallelization) {
          val (priority, target) = frontier.remove()
          if (target.children.nonEmpty) {
            target.children.foreach { child =>
              frontier.add((child.minDistance(pt.value), child))
            }
          } else {
            return (target, priority)
          }
        }
        Threading.parMap(frontier.asScala, threadpool)(priorityAndNode => {
          priorityAndNode._2.beamSearch(pt,beam = beamSize)
        }).minBy(_._2)
      }
    }
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                         Rotation Methods                          *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * Rotate this node. This essentially swaps the position of self's
    * sibling and self's aunt in the tree.
    *
    * Given:
    *   node = n
    *   node.sibling = s
    *   node.aunt = au
    *   node.grandparent = node.aunt.parent = gp
    *
    *  (1) Detach au from gp
    *  (2) Detach s and n from p
    *  (3) Create new internal node p' with au and n as children
    *  (4) Add p' as a child of gp
    *  (5) Add s as a child of gp
    */
  def rotate(): Unit = {
    val aunt = this.aunts().head
    val sibling = this.siblings().head
    val grandParent = this.parent.parent
    this.parent.isDeleted = true
    val newParent = new PerchNode(exactDistThreshold=this.exactDistThreshold,dim=this.dim)
    newParent.pts = null
    newParent.addChild(aunt)
    newParent.addChild(this)
    newParent.pointCounter = aunt.pointCounter + this.pointCounter
    grandParent.children.clear()
    grandParent.addChild(newParent)
    grandParent.addChild(sibling)
    this.parent.updateChildrenMinD()
    this.parent.updateChildrenMaxD()
    newParent.update()
  }

  /**
    * Rotate recursively if masking detected.
    * Check if the current node is masked and if so rotate. Recursively apply
    * this check to each node in the tree with early stopping if no masking
    * is detected.
    *
    * @param collapsibles (Optional) PriorityQueue storing the nodes which are next to be collapsed
    *                     when running in collapse mode.
    */
  def recursiveRotateIfMasked(collapsibles: PriorityQueue[(Double, PerchNode)] = null): Unit = {
    var currNode = this
    val rootNode = currNode.root()
    var masked = true
    while (currNode != rootNode && masked) {
      if (currNode.isMasked) {
        currNode.rotate()
        if (collapsibles != null && currNode.isLeaf && currNode.siblings()(0).isLeaf) {
          collapsibles.add((currNode.parent.childrenMaxD, currNode.parent))
        }
        currNode = currNode.parent
      } else if (currNode.children.nonEmpty) {
        masked = false
        currNode = currNode.parent
      } else {
        currNode = currNode.parent
      }
    }
  }

  /**
    * Rotate recursively if balance candidate detected.
    * Check if rotating the current node would increase balance in the tree if
    * rotated but would also not enduce any additional masking. If so, rotate.
    * Apply the check recursively up the tree.
    *
    * @param collapsibles (Optional) PriorityQueue storing the nodes which are next to be collapsed
    *                     when running in collapse mode.
    */
  def recursiveRotateIfUnbalanced(collapsibles: PriorityQueue[(Double, PerchNode)] = null): Unit = {
    var currNode = this
    val rootNode = currNode.root()
    while (currNode != rootNode) {
      val sibling = currNode.siblings().head
      val rotateOrder = if (sibling.pointCounter < currNode.pointCounter)
        IndexedSeq(sibling, currNode)
      else
        IndexedSeq(currNode, sibling)
      if (rotateOrder(0).canRotateForBalance) {
        rotateOrder(0).rotate()
        if (collapsibles != null && rotateOrder(0).isLeaf && rotateOrder(0).siblings()(0).isLeaf) {
          collapsibles.add((rotateOrder(0).parent.childrenMaxD, rotateOrder(0).parent))
        }
        currNode = rotateOrder(0).parent
      } else if (rotateOrder(1).canRotateForBalance) {
        rotateOrder(1).rotate()
        if (collapsibles != null && rotateOrder(1).isLeaf && rotateOrder(1).siblings()(0).isLeaf) {
          collapsibles.add((rotateOrder(1).parent.childrenMaxD, rotateOrder(1).parent))
        }
        currNode = rotateOrder(1).parent
      } else {
        currNode = currNode.parent
      }
    }
  }

  /**
    * Determine if this node is masked. From Equation 1,
    *
    * A node v with sibling v' and aunt a in a tree T is masked
    * if there exists a point x \in lvs(v) such that
    *      max ||x−y||   >    min ||x−z||
    *    y in lvs(v′)       z in lvs(a)
    *
    * We lower bound the maximum on the left hand side of Equation 1 by the minimum, which we then
    * approximate with a minDistance computation (Equation 2), and we upper bound the minimum on
    * the right hand side by the maximum, which we upper bound with a maxDistance computation (Equation 3).
    *
    *
    * @return Whether or not the node is masked according to the described check.
    */
  def isMasked: Boolean = {
    if (this.parent != null && this.parent.parent != null) {
      assert(this.aunts().length == 1)
      val aunt = this.aunts().head
      val sibling = this.siblings().head

      // We can write a shortcut here that compares to aunt and if the upper
      // bound is small enough stop; otherwise check the other max distance
      val auntMaxDist = math.max(aunt.maxDistance(this.mins),aunt.maxDistance(this.maxes))
      val otherMaxDist = math.max(this.maxDistance(aunt.mins),this.maxDistance(aunt.maxes))
      val worstMaxDist = math.min(auntMaxDist,otherMaxDist)
      val siblingMinDist = this.parent.childrenMinD
      worstMaxDist < siblingMinDist
    } else {
      false
    }
  }

  /**
    * Determine if this node can be rotated to improve the balance of the
    * tree without hurting dendrogram purity on separated data.
    *
    * @return Whether or not the node can be rotated for balance
    */
  def canRotateForBalance: Boolean = {
    balanceCandidate && _canRotateForBalance
  }


  /**
    * Determine if rotating this node (for balance) will produce
    * imperfect dendrogram purity on separated data. This checks
    * if there exists a point x_i in lvs(node) such that x_i is closer
    * to a leaf of node.aunt than to some leaf of node.sibling
    *
    * @return True/false for this condition
    */
  def _canRotateForBalance: Boolean = {
    if (this.parent != null && this.parent.parent != null) {
      val aunt = this.aunts().head
      val sibMaxDist = this.parent.childrenMaxD
      val auntMinDist = math.min(aunt.minDistance(this.mins), aunt.minDistance(this.maxes))
      val otherMinDist = math.min(this.minDistance(aunt.mins), this.minDistance(aunt.maxes))
      val bestAuntMinDist = math.max(auntMinDist, otherMinDist)
      sibMaxDist > bestAuntMinDist
    } else {
      false
    }
  }

  /**
    * Determine if the node can be rotated to improve the balance
    * of the tree.
    *
    * @return True/false for this condition
    */
  def balanceCandidate: Boolean = {
    if (this.parent != null && this.parent.parent != null) {
      assert(this.aunts().length == 1)
      val auntSize = this.aunts().head.pointCounter
      val parentSize = this.parent.pointCounter
      math.abs(parentSize - auntSize - 2*this.pointCounter) < math.abs(parentSize - auntSize)
    } else {
      false
    }
  }


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                         Insertion Methods                         *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * Insert a new point into the tree using A* exact nearest neighbor search.
    * Apply recurse masking and balance rotations and collapsing where appropriate.
    *
    * @param pt The point to be added
    * @param collapsibles (Optional) Priority queue of the nodes to be collapsed or null if not running
    *                     in collapsed mode. Default: null
    * @param L (Optional) Maximum number of leaves for collapsed mode. Default: no limit.
    * @return The root of the tree after insertion.
    */
  def insert(pt: Point, collapsibles: PriorityQueue[(Double, PerchNode)] = null,
             L: Int = Integer.MAX_VALUE): PerchNode = {
    assert(this.isRoot, "Points must be inserted at the root")
    if (this.pointCounter == 0) {
      this.addPt(pt)
      this.updateParamsRecursively()
      this
    } else {
      val (currNode,_) = this.AStar(pt)
      val newLeaf = currNode.splitDown(pt)
      val ancs = newLeaf.parent.ancestors()
      ancs.foreach{
        a =>
          a.addPt(pt)
      }
      newLeaf.updateParamsRecursively()

      if (collapsibles != null) {
        collapsibles.add((newLeaf.parent.childrenMaxD, newLeaf.parent))
      }

      newLeaf.siblings().head.recursiveRotateIfMasked(collapsibles = collapsibles)
      newLeaf.siblings().head.recursiveRotateIfUnbalanced(collapsibles = collapsibles)


      if (collapsibles != null && this.root().pointCounter > L) {
        var priorityAndBest = collapsibles.remove()
        var priority = priorityAndBest._1
        var best = priorityAndBest._2
        var valid = best.validCollapse()
        var upToDate = priority == best.childrenMaxD
        while (!valid || !upToDate) {
          if (!upToDate)
            collapsibles.add((best.childrenMaxD, best))
          priorityAndBest = collapsibles.remove()
          priority = priorityAndBest._1
          best = priorityAndBest._2
          valid = best.validCollapse()
          upToDate = priority == best.childrenMaxD
        }
        assert(priority == best.childrenMaxD)
        best.collapse()
        if (best.siblings()(0).isLeaf) {
          collapsibles.add((best.parent.childrenMaxD, best.parent))
        }
      }
      newLeaf.root()
    }
  }


  /**
    * Insert a new point into the tree using parallelized A* exact nearest neighbor search.
    * Apply recurse masking and balance rotations and collapsing where appropriate.
    *
    * @param pt The point to be added
    * @param collapsibles (Optional) Priority queue of the nodes to be collapsed or null if not running
    *                     in collapsed mode. Default: null
    * @param L (Optional) Maximum number of leaves for collapsed mode. Default: no limit.
    * @param maxFrontierSizeBeforeParallelization The maximum size of the frontier to build before searching in parallel
    *                                             (should be greater than number of threads used) (default 50)
    * @param threadpool The threadpool to use for parrallelism
    * @return The root after the insertion
    */
  def insertParallel(pt: Point, numThreads: Int,
                     collapsibles: PriorityQueue[(Double, PerchNode)] = null,
                     L: Int = Integer.MAX_VALUE, maxFrontierSizeBeforeParallelization: Int = 8)(implicit threadpool: ExecutorService): PerchNode = {
    assert(this.isRoot, "Points must be inserted at the root")
    if (this.pointCounter == 0) {
      this.addPt(pt)
      this.updateParamsRecursively()
      this
    } else {
      val (currNode,_) = this.AStarParallel(pt,maxFrontierSizeBeforeParallelization)
      val newLeaf = currNode.splitDown(pt)
      val ancs = newLeaf.parent.ancestors()
      ancs.foreach{
        a =>
          a.addPt(pt)
      }
      newLeaf.updateParamsRecursively()

      if (collapsibles != null) {
        collapsibles.add((newLeaf.parent.childrenMaxD, newLeaf.parent))
      }

      newLeaf.siblings().head.recursiveRotateIfMasked(collapsibles = collapsibles)
      newLeaf.siblings().head.recursiveRotateIfUnbalanced(collapsibles = collapsibles)

      if (collapsibles != null && this.root().pointCounter > L) {
        var priorityAndBest = collapsibles.remove()
        var priority = priorityAndBest._1
        var best = priorityAndBest._2
        var valid = best.validCollapse()
        var upToDate = priority == best.childrenMaxD
        while (!valid || !upToDate) {
          if (!upToDate)
            collapsibles.add((best.childrenMaxD, best))
          priorityAndBest = collapsibles.remove()
          priority = priorityAndBest._1
          best = priorityAndBest._2
          valid = best.validCollapse()
          upToDate = priority == best.childrenMaxD
        }
        assert(priority == best.childrenMaxD)
        //        println(best.childrenMaxD)
        best.collapse()
        if (best.siblings()(0).isLeaf) {
          collapsibles.add((best.parent.childrenMaxD, best.parent))
        }
      }
      newLeaf.root()
    }
  }


  /**
    * Insert a new point into the tree using beam search nearest neighbor search.
    * Apply recurse masking and balance rotations and collapsing where appropriate.
    *
    * @param pt The point to be added
    * @param collapsibles (Optional) Priority queue of the nodes to be collapsed or null if not running
    *                     in collapsed mode. Default: null
    * @param L (Optional) Maximum number of leaves for collapsed mode. Default: no limit.
    * @return The root of the tree after insertion.
    */
  def insertBeamSearch(pt: Point, collapsibles: PriorityQueue[(Double, PerchNode)] = null,
                       L: Int = Integer.MAX_VALUE, beam: Int = 100): PerchNode = {
    assert(this.isRoot, "Points must be inserted at the root")
    if (this.pointCounter == 0) {
      this.addPt(pt)
      this.updateParamsRecursively()
      this
    } else {
      val (currNode,_) = this.beamSearch(pt,beam)
      val newLeaf = currNode.splitDown(pt)
      val ancs = newLeaf.parent.ancestors()
      ancs.foreach{
        a =>
          a.addPt(pt)
      }
      newLeaf.updateParamsRecursively()

      if (collapsibles != null) {
        collapsibles.add((newLeaf.parent.childrenMaxD, newLeaf.parent))
      }

      newLeaf.siblings().head.recursiveRotateIfMasked(collapsibles = collapsibles)
      newLeaf.siblings().head.recursiveRotateIfUnbalanced(collapsibles = collapsibles)


      if (collapsibles != null && this.root().pointCounter > L) {
        var priorityAndBest = collapsibles.remove()
        var priority = priorityAndBest._1
        var best = priorityAndBest._2
        var valid = best.validCollapse()
        var upToDate = priority == best.childrenMaxD
        while (!valid || !upToDate) {
          if (!upToDate)
            collapsibles.add((best.childrenMaxD, best))
          priorityAndBest = collapsibles.remove()
          priority = priorityAndBest._1
          best = priorityAndBest._2
          valid = best.validCollapse()
          upToDate = priority == best.childrenMaxD
        }
        assert(priority == best.childrenMaxD)
        best.collapse()
        if (best.siblings()(0).isLeaf) {
          collapsibles.add((best.parent.childrenMaxD, best.parent))
        }
      }
      newLeaf.root()
    }
  }

  /**
    * Insert a new point into the tree using parallelized beam search nearest neighbor search.
    * Apply recurse masking and balance rotations and collapsing where appropriate.
    *
    * @param pt The point to be added
    * @param numThreads The number of threads
    * @param beam The beam width (per thread).  Multiply this value by numThreads for comparable value to single threaded version.
    * @param collapsibles (Optional) Priority queue of the nodes to be collapsed or null if not running
    *                     in collapsed mode. Default: null
    * @param L (Optional) Maximum number of leaves for collapsed mode. Default: no limit.
    * @param maxFrontierSizeBeforeParallelization The maximum size of the frontier to build before searching in parallel
    *                                             (should be greater than number of threads used) (default 50)
    *                                             Note that the search is done exactly with A* to find this frontier.
    * @param threadpool The threadpool to use.
    * @return The root after insertion
    */
  def insertBeamSearchParallel(pt: Point, numThreads: Int, beam: Int,
                               collapsibles: PriorityQueue[(Double, PerchNode)] = null,
                               L: Int = Integer.MAX_VALUE, maxFrontierSizeBeforeParallelization: Int = 50)(implicit threadpool: ExecutorService): PerchNode = {
    assert(this.isRoot, "Points must be inserted at the root")
    if (this.pointCounter == 0) {
      this.addPt(pt)
      this.updateParamsRecursively()
      this
    } else {
      val (currNode,_) = this.beamSearchParallel(pt,beam,numThreads,maxFrontierSizeBeforeParallelization)
      val newLeaf = currNode.splitDown(pt)
      val ancs = newLeaf.parent.ancestors()
      ancs.foreach{
        a =>
          a.addPt(pt)
      }
      newLeaf.updateParamsRecursively()

      if (collapsibles != null) {
        collapsibles.add((newLeaf.parent.childrenMaxD, newLeaf.parent))
      }

      newLeaf.siblings().head.recursiveRotateIfMasked(collapsibles = collapsibles)
      newLeaf.siblings().head.recursiveRotateIfUnbalanced(collapsibles = collapsibles)

      if (collapsibles != null && this.root().pointCounter > L) {
        var priorityAndBest = collapsibles.remove()
        var priority = priorityAndBest._1
        var best = priorityAndBest._2
        var valid = best.validCollapse()
        var upToDate = priority == best.childrenMaxD
        while (!valid || !upToDate) {
          if (!upToDate)
            collapsibles.add((best.childrenMaxD, best))
          priorityAndBest = collapsibles.remove()
          priority = priorityAndBest._1
          best = priorityAndBest._2
          valid = best.validCollapse()
          upToDate = priority == best.childrenMaxD
        }
        assert(priority == best.childrenMaxD)
        //        println(best.childrenMaxD)
        best.collapse()
        if (best.siblings()(0).isLeaf) {
          collapsibles.add((best.parent.childrenMaxD, best.parent))
        }
      }
      newLeaf.root()
    }
  }


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                          Update Methods                           *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * Update self's bounding box and determine if ancestors need update.
    *
    * Check if self's children have changed their bounding boxes. If not,
    * we're done. If they have changed, update this node's bounding box. Also,
    * determine whether this node needs to store points or not (based on the
    * exact distance threshold). There are a handful of scenarios here where
    * we must re-cache the distance at the parent and grandparent.
    * If this node has no children, update its bounding box and its parent's
    * cached distances.
    * @return A tuple of this node and a bool that is true if the parent may need an
    *         update.
    */
  def update(): (PerchNode, Boolean) = {
    if (this.children.nonEmpty) {
      val oldMins = this.mins
      val oldMaxes = this.maxes
      this.mins = elementwiseMin(this.children(0).mins, this.children(1).mins)
      this.maxes = elementwiseMax(this.children(0).maxes, this.children(1).maxes)
      if (this.pts == null) {
        val childPts = this.children.map(_.pointCounter).sum //this.children.foldLeft(0)((s, n) => s + n.pointCounter)
        if (childPts > 0 && childPts <= this.exactDistThreshold) {
          this.pts = new ArrayBuffer[Point](childPts)
          this.children.foreach(child => child.pts.foreach(pt => this.pts.append(pt)))
        }
      }
      val newMinsOrMaxes = !oldMins.sameElements(this.mins) || !oldMaxes.sameElements(this.maxes)
      val stillHavePts = this.pts != null || (this.parent != null && this.siblings()(0).pts != null)

      if (this.pts != null) {
        assert(this.pointCounter <= this.exactDistThreshold,s"${this.pts}\t${this.pointCounter}")
      }
      if (this.parent != null && this.siblings()(0).pts != null) {
        assert(this.siblings()(0).pointCounter <= this.exactDistThreshold)
      }

      if (this.parent != null) {
        this.parent.updateChildrenMinD()
        this.parent.updateChildrenMaxD()
      }
      assert(this.pts == null || this.pts.length == this.pointCounter, s"${this.pts}\t${this.pointCounter}")
      (this, newMinsOrMaxes || stillHavePts)
    } else {
      this.mins = this.pts(0).value
      this.maxes = this.pts(0).value
      if (this.parent != null) {
        this.parent.updateChildrenMinD()
        this.parent.updateChildrenMaxD()
      }
      assert(this.pts != null || this.pts.length == this.pointCounter)
      (this, true)
    }
  }

  /**
    * Update a node's parameters recursively.
    */
  def updateParamsRecursively(): Unit = {
    var (_, needUpdate) = this.update()
    var currNode = this
    while (currNode.parent != null && needUpdate) {
      needUpdate = currNode.parent.update()._2
      currNode = currNode.parent
    }
  }

  /**
    * Update the childrenMinD parameter.
    *
    * This parameter is a cached computation of the approximate min distance
    * between self's children. Find this distance by computing the min dist
    * between child1 and child2 and then from child2 to child1 (because it's
    * not a symmetric approximation) and taking the max to get the largest
    * lower bound.
    */
  def updateChildrenMinD(): Unit = {
    if (this.children.nonEmpty) {
      val c0 = this.children(0)
      val c1 = this.children(1)
      this.childrenMinD = math.max(
        math.min(c0.minDistance(c1.mins), c0.minDistance(c1.maxes)),
        math.min(c1.minDistance(c0.mins), c1.minDistance(c0.maxes)))
    }
  }

  /**
    * Update the childrenMaxD parameter.
    *
    * This parameter is a cached computation of the approximate max distance
    * between self.children. I find this distance by computing the max dist
    * between child1 and child2 and then from child2 to child1 (because it's
    * not a symmetric approximation) and taking the min to get the smallest
    * valid lower bound.
    */
  def updateChildrenMaxD(): Unit = {
    if (this.children.nonEmpty) {
      val c0 = this.children(0)
      val c1 = this.children(1)
      this.childrenMaxD = math.min(
        math.max(c0.maxDistance(c1.mins), c0.maxDistance(c1.maxes)),
        math.max(c1.maxDistance(c0.mins), c1.maxDistance(c0.maxes)))
    }
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                       Collapse Methods                            *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * Collapse this node, removing all of the structure beneath this node.
    * Moving all descendant leaves into a special state where their parent
    * is this node, but are inaccessible when adding new points.
    */
  def collapse(): Unit = {
    this.collapsedLeaves ++= this.leaves()
    this.collapsedLeaves.foreach(n => n.parent = this)
    // free up the memory that is being taken up by the full vectors.
    //    this.collapsedLeaves.foreach(n => n.pts.foreach(p => p.value = null))
    this.isCollapsed = true
    this.children.clear()
    assert(this.children.isEmpty)
  }

  /**
    * Determine if this node is allowed to be collapsed.
    * @return True/false if the node can be collapsed
    */
  def validCollapse(): Boolean = {
    !this.isDeleted && this.children.nonEmpty && this.children(0).isLeaf && this.children(1).isLeaf
  }

  /**
    * Find all of the nodes in the tree which can be collapsed.
    * This is used when collapsing is done as a post-processing
    * procedure rather than running Perch in collapsed mode.
    * @return
    */
  def findCollapsibles(): PriorityQueue[(Double, PerchNode)] = {
    val coll = this.leaves().map(_.parent).toSet[PerchNode]
    val pq = new PriorityQueue[(Double,PerchNode)](PQComparator)
    coll.foreach {
      c =>
        pq.add((c.childrenMaxD,c))
    }
    pq
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *            Extracting a Flat Clustering Methods                   *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */


  /**
    * Select a clustering by consecutively merging collapsible nodes based on
    * minimum number of points in the nodes.
    *
    * @param collapsibles The collapsible frontier
    * @param K The number of clusters to extract
    * @return This node
    */
  def pickKPointCounter(collapsibles: PriorityQueue[(Double, PerchNode)], K: Int): PerchNode = {
    assert(collapsibles != null)
    assert(K > 0)

    val newHeap = new PriorityQueue[(Double, PerchNode)](PQComparator)
    while (!collapsibles.isEmpty) {
      val (_, bcd) = collapsibles.remove()
      if (bcd.validCollapse()) {
        newHeap.add((bcd.pointCounter, bcd))
      }
    }
    var l = this.clusters().length
    while (l > K) {
      var best = newHeap.remove()._2
      while (!best.validCollapse()) {
        best = newHeap.remove()._2
      }
      best.collapse()
      if (best.siblings()(0).isLeaf) {
        newHeap.add((best.parent.pointCounter, best.parent))
      }
      l -= 1
    }
    this
  }

  /**
    * Select a clustering by consecutively merging collapsible nodes based on
    * minimum distance between the node's children.
    *
    * @param collapsibles The collapsible frontier
    * @param K The number of clusters to extract
    * @return This node
    */
  def pickKMinD(collapsibles: PriorityQueue[(Double, PerchNode)], K: Int): PerchNode = {
    assert(collapsibles != null)
    assert(K > 0)

    val newHeap = new PriorityQueue[(Double, PerchNode)](PQComparator)
    while (!collapsibles.isEmpty) {
      val (_, bcd) = collapsibles.remove()
      if (bcd.validCollapse()) {
        newHeap.add((bcd.childrenMinD, bcd))
      }
    }
    var l = this.clusters().length
    while (l > K) {
      var best = newHeap.remove()._2
      while (!best.validCollapse()) {
        best = newHeap.remove()._2
      }
      best.collapse()
      if (best.siblings()(0).isLeaf) {
        newHeap.add((best.parent.childrenMinD, best.parent))
      }
      l -= 1
    }
    this
  }

  /**
    * Select a clustering by consecutively merging collapsible nodes based on
    * maximum distance between the node's children.
    *
    * @param collapsibles The collapsible frontier
    * @param K The number of clusters to extract
    * @return This node
    */
  def pickKMaxD(collapsibles: PriorityQueue[(Double, PerchNode)], K: Int): PerchNode = {
    assert(collapsibles != null)
    assert(K > 0)

    val newHeap = new PriorityQueue[(Double, PerchNode)](PQComparator)
    while (!collapsibles.isEmpty) {
      val (_, bcd) = collapsibles.remove()
      if (bcd.validCollapse()) {
        newHeap.add((bcd.childrenMaxD, bcd))
      }
    }
    var l = this.clusters().length
    while (l > K) {
      var best = newHeap.remove()._2
      while (!best.validCollapse()) {
        best = newHeap.remove()._2
      }
      best.collapse()
      if (best.siblings()(0).isLeaf) {
        newHeap.add((best.parent.childrenMaxD, best.parent))
      }
      l -= 1
    }
    this
  }

  /**
    * Select a clustering by consecutively merging collapsible nodes based on
    * one half of the maximum distance times the number of points in the node
    * as an approximation of a kmeans style cost.
    *
    * @param collapsibles The collapsible frontier
    * @param K The number of clusters to extract
    * @return This node
    */
  def pickKApproxKMeansCost(collapsibles: PriorityQueue[(Double, PerchNode)], K: Int): PerchNode = {
    assert(collapsibles != null)
    assert(K > 0)

    val newHeap = new PriorityQueue[(Double, PerchNode)](PQComparator)
    while (!collapsibles.isEmpty) {
      val (_, bcd) = collapsibles.remove()
      if (bcd.validCollapse()) {
        newHeap.add((bcd.childrenMaxD*0.5*bcd.pointCounter, bcd))
      }
    }
    var l = this.clusters().length
    while (l > K) {
      var best = newHeap.remove()._2
      while (!best.validCollapse()) {
        best = newHeap.remove()._2
      }
      best.collapse()
      if (best.siblings()(0).isLeaf) {
        newHeap.add((best.parent.childrenMaxD*0.5*best.pointCounter, best.parent))
      }
      l -= 1
    }
    this
  }


  /**
    * Select a clustering by consecutively merging collapsible nodes based on
    * the kmeans cost of the node.
    *
    * @param collapsibles The collapsible frontier
    * @param K The number of clusters to extract
    * @return This node
    */
  def pickKLocalKmeans(collapsibles: PriorityQueue[(Double, PerchNode)], K: Int): PerchNode = {
    assert(collapsibles != null)
    assert(K > 0)

    val means = JavaHashMap[PerchNode,Array[Double]](this.pointCounter*2)

    def mean(bCDNode: PerchNode) = {
      if (means.contains(bCDNode))
        means(bCDNode)
      else {
        val pts = bCDNode.leaves()
        val mu = Array.fill(pts.head.maxes.length)(0.0)
        var n = 0
        pts.foreach{
          p =>
            plusEq(mu, p.pts.head.value)
            n += 1
        }
        divN(mu,n)
        means.put(bCDNode,mu)
        mu
      }
    }

    def kmeansCost(bcd: PerchNode) = {
      val myMean = mean(bcd)
      val cost = bcd.leaves().map(l => minusNorm(myMean,l.pts.head.value)).sum
      cost
    }

    val newHeap = new PriorityQueue[(Double, PerchNode)](PQComparator)
    while (!collapsibles.isEmpty) {
      val (_, bcd) = collapsibles.remove()
      if (bcd.validCollapse()) {
        newHeap.add((kmeansCost(bcd), bcd))
      }
    }
    var l = this.clusters().length
    while (l > K) {
      var best = newHeap.remove()._2
      while (!best.validCollapse()) {
        best = newHeap.remove()._2
      }
      best.collapse()
      if (best.siblings()(0).isLeaf) {
        newHeap.add((kmeansCost(best.parent), best.parent))
      }
      l -= 1
    }
    this
  }


  /**
    * Select a clustering by consecutively merging collapsible nodes based on
    * the difference between the kmeans cost of this node and the costs of it's children.
    *
    * @param collapsibles The collapsible frontier
    * @param K The number of clusters to extract
    * @return This node
    */
  def pickKGlobalKmeans(collapsibles: PriorityQueue[(Double, PerchNode)], K: Int): PerchNode = {
    assert(collapsibles != null)
    assert(K > 0)

    val means = JavaHashMap[PerchNode,Array[Double]](this.pointCounter*2)

    def mean(bCDNode: PerchNode) = {
      if (means.contains(bCDNode))
        means(bCDNode)
      else {
        val pts = bCDNode.leaves()
        val mu = Array.fill(pts.head.maxes.length)(0.0)
        var n = 0
        pts.foreach{
          p =>
            plusEq(mu, p.pts.head.value)
            n += 1
        }
        divN(mu,n)
        means.put(bCDNode,mu)
        mu
      }
    }

    def kmeansCost(bcd: PerchNode) = {
      val myMean = mean(bcd)
      val cost = bcd.leaves().map(l => minusNorm(myMean,l.pts.head.value)).sum
      val child1Mean = mean(bcd.children(0))
      val child2Mean = mean(bcd.children(1))
      val child1Cost = bcd.children(0).leaves().map(l => minusNorm(child1Mean,l.pts.head.value)).sum
      val child2Cost = bcd.children(1).leaves().map(l => minusNorm(child2Mean,l.pts.head.value)).sum
      cost - child1Cost - child2Cost
    }

    val newHeap = new PriorityQueue[(Double, PerchNode)](PQComparator)
    while (!collapsibles.isEmpty) {
      val (_, bcd) = collapsibles.remove()
      if (bcd.validCollapse()) {
        newHeap.add((kmeansCost(bcd), bcd))
      }
    }
    var l = this.clusters().length
    while (l > K) {
      var best = newHeap.remove()._2
      while (!best.validCollapse()) {
        best = newHeap.remove()._2
      }
      best.collapse()
      if (best.siblings()(0).isLeaf) {
        newHeap.add((kmeansCost(best.parent), best.parent))
      }
      l -= 1
    }
    this
  }

  /**
    * Return the selected flat clustering from this tree.
    * @return
    */
  def clusters(): Seq[PerchNode] = {
    val lvs = new ArrayBuffer[PerchNode]()
    val queue = new mutable.Queue[PerchNode]()
    queue.enqueue(this)
    while (queue.nonEmpty) {
      val n = queue.dequeue()
      if (n.children.nonEmpty) {
        n.children.foreach(child => queue.enqueue(child))
      } else {
        lvs += n
      }
    }
    lvs
  }

  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                  Tree Structure Methods                           *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * Add newChild to this node's children
    * @param newChild The child to add
    * @return This node
    */
  def addChild(newChild: PerchNode): PerchNode = {
    newChild.parent = this
    this.children += newChild
    this
  }

  /**
    * Add pt as a point stored in this node.
    * (Does not update the node's bounding box)
    * @param pt The point to add
    */
  def addPt(pt: Point): Unit = {
    this.pointCounter += 1
    if (this.pts != null && this.pointCounter > this.exactDistThreshold) {
      this.pts = null
    } else if (this.pts != null) {
      this.pts.append(pt)
    }
  }

  /**
    * The siblings of this node
    * @return Sequence of siblings
    */
  def siblings(): Seq[PerchNode] = {
    if (this.parent != null)
      this.parent.children.filterNot(_ == this)
    else
      Seq()
  }

  /**
    * The aunts of this node
    * @return Sequence of aunts
    */
  def aunts(): Seq[PerchNode] = {
    if (this.parent != null && this.parent.parent != null) {
      this.parent.parent.children.filterNot(_ == this.parent)
    } else
      Seq()
  }

  /**
    * Perform a Split operation on this node.
    *
    * The Split operation is:
    *  - Create a new node, n', storing the input point pt
    *  - Detach this node from it's current parent,p.
    *  - Create a new node, p', which has n' and this node as children
    *  - Add p' as a child of p
    *
    * @param pt The point to be added into a new node as described above
    * @return The new node containing pt.
    */
  def splitDown(pt: Point): PerchNode = {
    val newInternal = new PerchNode(exactDistThreshold=this.exactDistThreshold,dim=this.dim)
    if (this.pts != null)
      newInternal.pts ++= this.pts
    else
      newInternal.pts = null
    newInternal.pointCounter = this.pointCounter
    if (this.parent != null) {
      this.parent.addChild(newInternal)
      this.parent.children.-=(this)
      newInternal.addChild(this)
    } else {
      newInternal.addChild(this)
    }
    val newLeaf =  new PerchNode(exactDistThreshold=this.exactDistThreshold,dim=this.dim)
    newLeaf.addPt(pt)
    newInternal.addChild(newLeaf)
    newInternal.addPt(pt)
    newLeaf
  }

  /**
    * Find all of the ancestors of this node.
    * @return Sequence of ancestors
    */
  def ancestors(): Seq[PerchNode] = {
    val anc = new ArrayBuffer[PerchNode](100) // todo: better estimate of initial size
    var curr = this
    while (curr.parent != null) {
      anc += curr.parent
      curr = curr.parent
    }
    anc
  }

  /**
    * Find all of the descendant leaves of this node.
    * @return Sequence of leaves
    */
  def leaves(): Seq[PerchNode] = {
    val lvs = new ArrayBuffer[PerchNode](this.pointCounter)
    val queue = new mutable.Queue[PerchNode]()
    queue.enqueue(this)
    while (queue.nonEmpty) {
      val n = queue.dequeue()
      if (n.collapsedLeaves.nonEmpty) {
        assert(n.children.isEmpty)
        lvs ++= n.collapsedLeaves
      }
      else if (n.isLeaf)
        lvs += n
      else
        n.children.foreach{
          c =>
            queue.enqueue(c)
        }
    }
    lvs
  }

  /**
    * The root of this Perch cluster tree
    * @return The root
    */
  def root(): PerchNode = {
    var currNode = this
    while (!currNode.isRoot)
      currNode = currNode.parent
    currNode
  }

  /**
    * Determine if this nodes is a leaf in the tree
    * @return True/false if the node is a leaf
    */
  def isLeaf: Boolean = this.children.isEmpty

  /**
    * Determine if this node is the root of the tree
    * @return True/false if the node is the root
    */
  def isRoot: Boolean = this.parent == null


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
   *                          Util Methods                             *
   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /**
    * Break any ordering ties by using the id numbers of the nodes.
    * @param o Another perch node
    * @return Comparison by id
    */
  override def compareTo(o: PerchNode): Int = this.id.compareTo(o.id)

  /**
    * Euclidean distance between x and y, ||x-y||
    * @param x Vector 1
    * @param y Vector 2
    * @return Distance
    */
  def minusNorm(x: Array[Double],y: Array[Double]): Double = {
    ComparisonCounter.increment()
    var i = 0
    val len = x.length
    assert(y.length == len)
    var res = 0.0
    while (i < len) {
      val r = x(i) - y(i)
      res += r*r
      i += 1
    }
    math.sqrt(res)
  }

  /**
    * Take the element-wise min between x and y
    * @param x Vector 1
    * @param y Vector 2
    * @return Elementwise min
    */
  def elementwiseMin(x: Array[Double],y: Array[Double]): Array[Double] = {
    val min = new Array[Double](x.length)
    var i = 0
    val len = x.length
    while (i < len) {
      min(i) = math.min(x(i),y(i))
      i += 1
    }
    min
  }

  /**
    * Take the element-wise max between x and y
    * @param x Vector 1
    * @param y Vector 2
    * @return Elementwise min
    */
  def elementwiseMax(x: Array[Double],y: Array[Double]): Array[Double] = {
    val max = new Array[Double](x.length)
    var i = 0
    val len = x.length
    while (i < len) {
      max(i) = math.max(x(i),y(i))
      i += 1
    }
    max
  }

  /**
    * Perform the elementwise addition left += right
    * @param left vector to be added to
    * @param right vector to add
    */
  def plusEq(left: Array[Double],right: Array[Double]) = {
    var i = 0
    val len = left.length
    while (i < len) {
      left(i) += right(i)
      i += 1
    }
  }

  /**
    * Performt the elementwise division left /= n
    * @param left the vector to be divided
    * @param n the scalar to divide by
    */
  def divN(left: Array[Double],n: Double) = {
    var i = 0
    val len = left.length
    while (i < len) {
      left(i) /= n
      i += 1
    }
  }


  /**
    * Write the tree in the evaluation file format:
    * Node Id \t Parent Id \t Label
    *
    * ParentId will be "None" if the node is the root
    * Label will be "None" for every internal node
    * and the ground truth class label for leaf nodes.
    * @param file The file to write to
    */
  def serializeTree(file: File): Unit = {
    val pw = new PrintWriter(file,"UTF-8")
    val queue = new scala.collection.mutable.Queue[PerchNode]()
    queue.enqueue(this)
    var currNode = this
    while (queue.nonEmpty) {
      currNode = queue.dequeue()
      val nodeId = if (currNode.isLeaf && currNode.collapsedLeaves.isEmpty) currNode.pts.head.pid else currNode.id.toString
      val parId = if (currNode.parent != null) currNode.parent.id else "None"
      val pid = if (currNode.isLeaf && currNode.collapsedLeaves.isEmpty) currNode.pts.head.label else "None"
      pw.println(s"$nodeId\t$parId\t$pid")
      currNode.children.foreach{
        c =>
          queue.enqueue(c)
      }
      currNode.collapsedLeaves.foreach{
        c =>
          queue.enqueue(c)
      }
    }
    pw.close()
  }

}

/**
  * Comparator used for various priorities in PerchNode class
  */
object PQComparator extends Comparator[(Double,PerchNode)] {
  override def compare(o1: (Double, PerchNode), o2: (Double, PerchNode)): Int = Ordering.Tuple2[Double,PerchNode].compare(o1,o2)
}