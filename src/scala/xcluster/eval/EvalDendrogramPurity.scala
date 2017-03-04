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

package xcluster.eval

import java.io.File
import java.util

import cc.factorie._
import cc.factorie.util.{DefaultCmdOptions, Threading}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import xcluster._


/**
  * Tree data structure used to compute dendrogram purity
  * @param id The unique id for this node
  * @param parent The parent node or None if the node is the root
  * @param parentId The id of the parent node or None if the node is the root
  * @param children The children objects of this node
  * @param labels The class lables of the descendant leaves of this node.
  */
class EvalTreeNode(
  val id: String,
  var parent: Option[EvalTreeNode],
  val parentId: Option[String],
  var children: ArrayBuffer[EvalTreeNode], var labels: ArrayBuffer[String]) {

  /**
    * Whether or not the node is a leaf
    * @return True/false if this is a leaf
    */
  def isLeaf: Boolean = this.children.isEmpty

  /**
    * Return all of the descendant leaves of this node
    * @return Iterable of nodes
    */
  def leaves(): Iterable[EvalTreeNode] = {
    if (this.isLeaf)
      Iterable(this)
    else {
      var curr_node = this
      val q = new mutable.Queue[EvalTreeNode]()
      q.enqueue(curr_node)
      val leaves = new ArrayBuffer[EvalTreeNode]()
      while (q.nonEmpty) {
        curr_node = q.dequeue()
        curr_node.children.foreach {
          c =>
            if (c.isLeaf)
              leaves += c
            else
              q.enqueue(c)
        }
      }
      leaves
    }
  }

  /**
    * Compute the purity of this node from the cached labels
    * @param wrt Class label about which purity should be computed
    * @return Purity value for the given class
    */
  def purity(wrt: String): Double = {
    // TODO: We could cache these scores
    labels.count(_ == wrt).toDouble / labels.size
  }

  /**
    * The purity of the most frequently appearing class in this node
    * @return Purity value for the given class
    */
  def purity(): Double = {
    val mostFreq = labels.groupBy(identity).mapValues(_.size).maxBy(_._2)._1
    purity(mostFreq)
  }

  /**
    * The ancestors of this node (does not include this node).
    * @return An iterable of nodes
    */
  def ancestors(): IndexedSeq[EvalTreeNode] = {
    val ancestors = new ArrayBuffer[EvalTreeNode]()
    var currNode = this
    while (currNode.parent.isDefined) {
      currNode = currNode.parent.get
      ancestors += currNode
    }
    ancestors
  }

  /**
    * The descendants of this node (does include this node)
    * @return The descendants of the node and this node itself
    */
  def descendants(): IndexedSeq[EvalTreeNode] = {
    val ds = new ArrayBuffer[EvalTreeNode]()
    val q = new mutable.Queue[EvalTreeNode]()
    q.enqueue(this)
    while (q.nonEmpty) {
      val target = q.dequeue()
      ds.append(target)
      target.children.foreach(q.enqueue(_))
    }
    ds.toIndexedSeq
  }

  /**
    * The root of the tree.
    * @return root
    */
  def root():EvalTreeNode = {
    var currNode = this
    while (currNode.parent.isDefined) {
      currNode = currNode.parent.get
    }
    currNode
  }


  /**
    * The least common ancestor of this node and the given other node
    * @param other A node in the tree
    * @return lca(this,other)
    */
  def lca(other: EvalTreeNode): EvalTreeNode = {
    val thisAncestors = this.ancestors()
    val otherAncestors = other.ancestors()
    var i = 0
    var found = false
    var lca = this.root()
    while (i < thisAncestors.length && !found) {
      var j = 0
      while(j < otherAncestors.length && !found) {
        if (thisAncestors(i) == otherAncestors(j)) {
          found = true
          lca = thisAncestors(i)
        }
        j += 1
      }
      i += 1
    }
    i-=1
    lca
  }

}

/**
  * Loaders for evaluation tree file format:
  * NodeId \t ParentId \t Label
  *
  * Parent and or label can be none or null if empty.
  */
object LoadEvalTree {

  /**
    * Load a serialized evaluation tree into memory and return the root of the tree
    * @param filename The filename
    */
  def load(filename: String): EvalTreeNode = {
    val forest = loadForest(filename)
    assert(forest.size == 1, s"File: $filename contained a forest of trees, not one tree.")
    forest.head
  }

  /**
    * Load a forest of evaluation trees into memory returning the roots of all the trees.
    * @param fn
    * @return
    */
  def loadForest(fn: String) = {

    val nodes = new java.util.HashMap[String,EvalTreeNode]().asScala

    new File(fn).lines("UTF-8").zipWithIndex.foreach {
      case (line,idx) =>
//        if (idx % 100 == 0)
//          println(s"Read $idx lines of $fn")
        val Array(id,parent,label) = line.split("\t")
        val parentId = if (parent.equalsIgnoreCase("none") || parent.equalsIgnoreCase("null")) None else Some(parent)
        val labels = new ArrayBuffer[String]()
        if (!(label.equalsIgnoreCase("none") || label.equalsIgnoreCase("null")))
          labels += label
        val node = new EvalTreeNode(id,None,parentId,new ArrayBuffer[EvalTreeNode](),labels)
        nodes.put(node.id,node)
    }
    nodes.values.foreach{
       n =>
        if (n.parentId.isDefined) {
          n.parent = nodes.get(n.parentId.get)
          if (n.parent.isEmpty)
            System.err.println(s"Missing ${n.parentId.get}")
          n.parent.get.children += n

        }
    }

    def propagateLabel(leaf: EvalTreeNode) = {
      val label = leaf.labels.head
      var currNode = leaf
      while (currNode.parent.isDefined) {
        currNode = currNode.parent.get
        currNode.labels += label
      }
    }
    val leaves = nodes.values.filter(_.isLeaf)
    leaves.foreach(propagateLabel)
    nodes.values.filter(_.parent.isEmpty)
  }


}

trait DendrogramPurityOpts extends DefaultCmdOptions {
  val input = new CmdOption[String]("input","the tree file to score")
  val algorithm = new CmdOption[String]("algorithm","Algorithm","STRING","the algorithm name")
  val dataset = new CmdOption[String]("dataset","dataset","STRING","the dataset name")
  val threads = new CmdOption[Int]("threads",4,"INT","number of threads to use")
  val print = new CmdOption[Boolean]("print",false,"BOOLEAN","print status updates for computation defaults to be false")
  val idFile = new CmdOption[String]("id-file","The file containing the point ids on which to evaluate dendrogram purity. Leave blank or as None to do exact dendrogram purity")
}

/**
  * Executable for dendrogram purity and expected
  * dendrogram purity. By default, exact dendrogram purity is computed
  * if an id-file is passed in then expected dendrogram
  * purity is computed on only those ids.
  */
object EvalDendrogramPurity {
  def main(args: Array[String]): Unit = {
    val opts = new DendrogramPurityOpts {}
    opts.parse(args)
    if (opts.idFile.wasInvoked && opts.idFile.value.toLowerCase != "none")
      ExpectedDendrogramPurity.run(opts)
    else
      DendrogramPurity.run(opts)
  }
}


object DendrogramPurity {

  /**
    * Find all pairs of points which have the same class label
    * @param root The root of the tree to evaluate
    * @param numThreads The number of threads to use
    * @return All pairs of points in dendrogram purity calculations
    */
  def allPairsForEval(root: EvalTreeNode, numThreads: Int): ArrayBuffer[(EvalTreeNode,EvalTreeNode)] = {
    val leaves = root.leaves()
    val byClass = new util.HashMap[String,ArrayBuffer[EvalTreeNode]]().asScala
    leaves.zipWithIndex.foreach{
      case (l,idx) =>
        if (idx % 100 == 0)
//          println(s"Processed $idx leaves")
          assert(l.labels.size == 1)
          if (!byClass.contains(l.labels.head))
            byClass.put(l.labels.head, new ArrayBuffer[EvalTreeNode]())
          byClass(l.labels.head) += l
    }
//    println("Finding pairs.")
    val allPairs = new ArrayBuffer[(EvalTreeNode,EvalTreeNode)](byClass.values.map(f => f.size * f.size).sum)
    byClass.values.foreach{
      classOnlyPoints =>
        if (classOnlyPoints.length > 100)
          allPairs ++= classOnlyPoints.pairsParallel(numThreads)
        else
          allPairs ++= classOnlyPoints.pairs
    }
    allPairs
  }

  /**
    * Compute the dendrogram purity for the given pairs
    * @param pairs Pairs of points to evaluate
    * @param threads Number of threads to use
    * @param print Whether or not to print status updates
    * @return Dendrogram purity
    */
  def evalPar(pairs: IndexedSeq[(EvalTreeNode,EvalTreeNode)],threads:Int,print:Boolean): Double = {
    val bufferSize = 10000
    val startIndexes = 0 until pairs.size by bufferSize
    @volatile var sum_purities = 0.0
    @volatile var N = 0.0
    Threading.parForeach(startIndexes,threads)({
      start =>
        var local_n = 0.0
        var local_purity = 0.0
        var i = start
        val end = math.min(start+ bufferSize,pairs.length)
        while (i < end) {
          val pair = pairs(i)
          local_purity += pair._1.lca(pair._2).purity(pair._1.labels.head)
          local_n += 1.0
          i += 1
        }
        synchronized{
          if (print)
            System.err.print(s"\rThread ${Thread.currentThread().getId} Computing purities for pairs ${start} to ${end} of ${pairs.size} = ${100*end/pairs.size.toFloat}% done")
          N += local_n
          sum_purities += local_purity
        }
    })
    sum_purities / N
  }

  /**
    * Run the main executable.
    * @param opts
    */
  def run(opts: DendrogramPurityOpts): Unit = {
    val root = LoadEvalTree.load(opts.input.value)
    val pairs = allPairsForEval(root,opts.threads.value)
    val score = evalPar(pairs,opts.threads.value,opts.print.value)
    println(s"${opts.algorithm.value}\t${opts.dataset.value}\t${score}")
  }

  def main(args: Array[String]): Unit = {
    val opts = new DendrogramPurityOpts {}
    opts.parse(args)
    run(opts)
  }


}

/**
  * Expected dendrogram computations
  */
object ExpectedDendrogramPurity {

  def main(args: Array[String]): Unit = {
    val opts = new DendrogramPurityOpts {}
    opts.parse(args)
    run(opts)
  }

  def run(opts: DendrogramPurityOpts) = {
    val ids = io.Source.fromFile(opts.idFile.value)("UTF-8").getLines().toSet[String]
    val root = LoadEvalTree.load(opts.input.value)
    val pairs = all_pairs_for_eval(root,ids,opts.threads.value)
    val score = evalPar(pairs,opts.threads.value,opts.print.value)
    println(s"${opts.algorithm.value}\t${opts.dataset.value}\t${score}")
  }

  /**
    * Compute all pairs for evaluation with respect to a given set of point ids
    * @param root the root node
    * @param idSet the ids
    * @param numThreads number of threads
    * @return
    */
  def all_pairs_for_eval(root: EvalTreeNode, idSet: Set[String],numThreads: Int): ArrayBuffer[(EvalTreeNode,EvalTreeNode)] = {
    val leaves = root.leaves()
    val byClass = new util.HashMap[String,ArrayBuffer[EvalTreeNode]]().asScala
    leaves.zipWithIndex.foreach{
      case (l,idx) =>
//        if (idx % 100 == 0)
//          println(s"Processed $idx leaves")
        if (idSet.contains(l.id)) {
          assert(l.labels.size == 1)
          if (!byClass.contains(l.labels.head))
            byClass.put(l.labels.head, new ArrayBuffer[EvalTreeNode]())
          byClass(l.labels.head) += l
        }
    }
//    println("Finding pairs.")
    val allPairs = new ArrayBuffer[(EvalTreeNode,EvalTreeNode)](byClass.values.map(f => f.size * f.size).sum)
    byClass.values.foreach{
      classOnlyPoints =>
        if (classOnlyPoints.length > 100)
          allPairs ++= classOnlyPoints.pairsParallel(numThreads)
        else
          allPairs ++= classOnlyPoints.pairs
    }
    allPairs
  }

  /**
    * Evaluate expected dendrogram purity
    * @param pairs the pairs to evaluate on
    * @param threads the number of threads
    * @param print whether or not to print status updates
    * @return
    */
  def evalPar(pairs: IndexedSeq[(EvalTreeNode,EvalTreeNode)],threads:Int,print:Boolean) = {
    val bufferSize = 1000
    val startIndexes = 0 until pairs.size by bufferSize
    @volatile var sum_purities = 0.0
    @volatile var N = 0.0
    Threading.parForeach(startIndexes,threads)({
      start =>
        var local_n = 0.0
        var local_purity = 0.0
        var i = start
        val end = math.min(start+ bufferSize,pairs.length)
        while (i < end) {
          val pair = pairs(i)
          local_purity += pair._1.lca(pair._2).purity(pair._1.labels.head)
          local_n += 1.0
          i += 1
        }
        synchronized{
          if (print)
            System.err.println(s"Thread ${Thread.currentThread().getId} Computing purities for pairs ${start} to ${end} of ${pairs.size} = ${100*end/pairs.size.toFloat}% done")
          N += local_n
          sum_purities += local_purity
        }
    })
    sum_purities / N
  }


}
