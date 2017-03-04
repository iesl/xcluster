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

import java.io.{File, PrintWriter}
import java.util.PriorityQueue

import cc.factorie.util.{DefaultCmdOptions, Threading}
import xcluster.models.{PQComparator, PerchNode, Point}
import xcluster.utils.ComparisonCounter
import xcluster._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random


trait RunPerchOpts extends DefaultCmdOptions {
  val input = new CmdOption[String]("input","The input file to cluster (Required).",true)
  val outDir = new CmdOption[String]("outdir","Where to write the experiment output (Required)",true)
  val algorithm = new CmdOption[String]("algorithm","Perch","STRING","The algorithm name to record in the results. Default: Perch")
  val datasetName = new CmdOption[String]("dataset","","STRING","The dataset name to record in the results. Default: input filename")
  val threads = new CmdOption[Int]("threads",24,"INT","The number of threads to use. Default: 24")
  val maxFrontierSizeBeforeParallelization = new CmdOption[Int]("max-frontier-par",50,"INT","The min points before invoking parallelization")
  val L = new CmdOption[String]("max-leaves", "None", "INT or None", "maximum number of leaves.  Default: None (no clustering extracted)")
  val K = new CmdOption[String]("clusters", "None", "INT or None", "The number of clusters. Default: None (no clustering extracted) ")
  val exactDistThreshold = new CmdOption[Int]("exact-dist-thres",10,"INT","The number of points to search using exact dist threshold.")
  val pickKMethod = new CmdOption[String]("pick-k","approxKM","STRING","the method used for picking k: pointCounter, maxD, minD, approxKM (default), globalKM, localKM")
  val beam = new CmdOption[String]("beam","None","STRING","The beam size or None to not use a beam. Default None")
  val countComparisons = new CmdOption[Boolean]("count-comparisons",false,"boolean", "Whether or not to count comparisons. Default: False")
  val quiet = new CmdOption[Boolean]("quiet",false,"boolean","Whether to skip printed status updates. Default: False")
}

/**
  * Run PERCH on a given dataset.
  */
object RunPerch {

  def main(args: Array[String]): Unit = {

    // set random seed
    implicit val rdom = new Random(17)


    // Parse command line arguments
    val opts = new RunPerchOpts {}
    opts.parse(args)

    println("Running EvalDataset")
    println("Command line arguments: ")
    opts.values.foreach(f => println(s"${f.name}: ${f.value}"))

    if (opts.countComparisons.value)
      ComparisonCounter.on()

    val outDir = new File(opts.outDir.value)
    outDir.mkdirs()

    var collapsibles = {
      if (opts.L.wasInvoked && opts.L.value.toLowerCase != "none" && opts.L.value.toInt > 0)
        new PriorityQueue[(Double, PerchNode)](PQComparator)
      else
        null
    }

    val L = {
      if (opts.L.wasInvoked && opts.L.value.toLowerCase != "none" && opts.L.value.toInt > 0) {
        opts.L.value.toInt
      } else {
        -1
      }
    }

    println(s"Collapsibles: ${collapsibles}")
    println(s"L: $L")

    val beQuiet = opts.quiet.value
    val threads = opts.threads.value
    val maxFrontierSizeBeforeParallelization = if (opts.maxFrontierSizeBeforeParallelization.wasInvoked) opts.maxFrontierSizeBeforeParallelization.value else opts.threads.value
    var root = new PerchNode(exactDistThreshold = opts.exactDistThreshold.value, dim = LoadPoints.loadFile(opts.input.value).next.value.length)
    val points = LoadPoints.loadFile(opts.input.value)
    val start = System.currentTimeMillis()


    // Process the dataset point at a time
    if (opts.threads.value == 1) {
      points.zipWithIndex.foreach {
        case (p, count) =>
          if (!beQuiet) {
            if (count % 100 == 0)
              println(s"Processed $count points in ${(System.currentTimeMillis() - start) / 1000.0} seconds. ${((System.currentTimeMillis() - start) / 1000.0) / (count + 1)} seconds per point")
          }
          if (opts.beam.value.toLowerCase == "none") {
            root = root.insert(p, collapsibles = collapsibles, L = L)
          }else {
            root = root.insertBeamSearch(p, collapsibles = collapsibles, L = L, beam = opts.beam.value.toInt)
          }
      }
    } else {
      implicit val threadPool = Threading.newFixedThreadPool(threads)
      points.zipWithIndex.foreach {
        case (p, count) =>
          if (!beQuiet) {
            if (count % 100 == 0)
              println(s"Processed $count points in ${(System.currentTimeMillis() - start) / 1000.0} seconds. ${((System.currentTimeMillis() - start) / 1000.0) / (count + 1)} seconds per point")
          }
          if (opts.beam.value.toLowerCase == "none")
            root = root.insertParallel(p, threads, collapsibles = collapsibles, L = L,maxFrontierSizeBeforeParallelization = maxFrontierSizeBeforeParallelization)
          else
              root = root.insertBeamSearchParallel(p, threads,opts.beam.value.toInt, collapsibles = collapsibles, L = L,maxFrontierSizeBeforeParallelization = maxFrontierSizeBeforeParallelization)
      }
      threadPool.shutdown()
    }
    val end = System.currentTimeMillis()
    val runningTimeSeconds = (end - start).toDouble / 1000.0

    // Write running time to a file
    val pw = new PrintWriter(new File(outDir,"running_time.txt"))
    val datasetName = if (opts.datasetName.wasInvoked) opts.datasetName.value else new File(opts.input.value).getName
    pw.println(s"${opts.algorithm.value}\t$datasetName\t$runningTimeSeconds")
    pw.close()

    // Write tree to the file
    root.serializeTree(new File(outDir,"tree.tsv"))

    // Select the clustering
    println(s"Number of clusters (before picking k) = ${root.clusters().size}")
    if (opts.K.wasInvoked && opts.K.value.toLowerCase != "none" && opts.K.value.toInt > 0) {
      if (collapsibles == null)
        collapsibles = root.findCollapsibles()
      if (opts.pickKMethod.value.toLowerCase == "pointcounter")
        root.pickKPointCounter(collapsibles = collapsibles, K = opts.K.value.toInt)
      else if (opts.pickKMethod.value.toLowerCase == "approxkm")
        root.pickKApproxKMeansCost(collapsibles = collapsibles, K = opts.K.value.toInt)
      else if (opts.pickKMethod.value.toLowerCase == "maxd")
        root.pickKMaxD(collapsibles = collapsibles, K = opts.K.value.toInt)
      else if (opts.pickKMethod.value.toLowerCase == "mind")
        root.pickKMinD(collapsibles = collapsibles, K = opts.K.value.toInt)
      else if (opts.pickKMethod.value.toLowerCase == "globalkm")
        root.pickKGlobalKmeans(collapsibles = collapsibles, K = opts.K.value.toInt)
      else if (opts.pickKMethod.value.toLowerCase == "localkm")
        root.pickKLocalKmeans(collapsibles = collapsibles, K = opts.K.value.toInt)
      else
        println(s"Unknown method for picking k: ${opts.pickKMethod.value}")
    }

    // Write the clustering
    val predicted = new ArrayBuffer[(String, String)]()
    val goldClustering = new ArrayBuffer[(String, String)]()
    println(s"Number of clusters (after picking k) = ${root.clusters().size}")
    var idx = 0
    var cIdx = 0
    root.clusters().foreach{ cluster =>
      cluster.leaves().flatMap(_.pts).foreach{ pt =>
        predicted.append((pt.pid, cIdx.toString))
        goldClustering.append((pt.pid, pt.label))
        idx += 1
      }
      cIdx += 1
    }
    val predictedPW = new PrintWriter(new File(outDir, "predicted.txt"), "UTF-8")
    predicted.foreach{ case (i1, i2) => predictedPW.write(s"$i1\t$i2\n") }
    predictedPW.close()

    val goldPW = new PrintWriter(new File(outDir, "gold.txt"), "UTF-8")
    goldClustering.foreach{ case (i1, i2) => goldPW.write(s"$i1\t$i2\n") }
    goldPW.close()

    // Print the number of comparisons
    val numberOfComparisons = ComparisonCounter.count
    val compPW = new PrintWriter(new File(outDir,"comparisons.txt"))
    compPW.println(s"${opts.algorithm.value}\t$datasetName\t${numberOfComparisons.get()}")
    compPW.close()
  }


}

/**
  * Load a file of points. The file format is one point per line as:
  * point_id \t gold_label \t vector (tab separated)
  */
object LoadPoints {
  def loadLine(line: String) = {
    val splt = line.split("\t")
    val pid = splt(0)
    val label = splt(1)
    val vec = splt.drop(2).map(_.toDouble)
    Point(pid,label,vec)
  }

  def loadFile(file: File): Iterator[Point] = file.lines("UTF-8").map(loadLine)

  def loadFile(filename: String): Iterator[Point] = loadFile(new File(filename))
}

/**
  * Load a file of weighted points. The file format is one point per line as:
  * weighted \t point_id \t gold_label \t vector (tab separated)
  */
object LoadWeightedPoints {
  def loadLine(line: String) = {
    val splt = line.split("\t")
    val weight = splt(0).toDouble
    val pid = splt(1)
    val label = splt(2)
    val vec = splt.drop(3).map(_.toDouble)
    Point(pid,label,vec,weight)
  }

  def loadFile(file: File): Iterator[Point] = file.lines("UTF-8").map(loadLine)

  def loadFile(filename: String): Iterator[Point] = loadFile(new File(filename))
}