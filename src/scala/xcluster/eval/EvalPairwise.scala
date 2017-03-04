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

import cc.factorie.util.ClusterF1Evaluation.Pairwise
import cc.factorie.util._
import xcluster._

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

trait EvalPairwiseOpts extends DefaultCmdOptions {
  val predicted = new CmdOption[String]("predicted","the predicted clustering file")
  val gold = new CmdOption[String]("gold","the gold clustering file")
  val algorithm = new CmdOption[String]("algorithm","Algorithm","STRING","the algorithm name")
  val dataset = new CmdOption[String]("dataset","dataset","STRING","the dataset name")
  val threads = new CmdOption[Int]("threads",4,"INT","number of threads to use")
  val print = new CmdOption[Boolean]("print",false,"BOOLEAN","print status updates for computation defaults to be false")
  val idFile = new CmdOption[String]("id-file","The file containing the point ids on which to evaluate pairwise prec/rec/f1. Leave blank or as None use all points")
}

/**
  * Run Pairwise Precision, Recall, and F1 evaluation for flat clustering
  */
object EvalPairwise {

  def main(args: Array[String]): Unit = {
    val opts = new EvalPairwiseOpts {}
    opts.parse(args)

    val ids = if (opts.idFile.wasInvoked && opts.idFile.value.toLowerCase != "none") new File(opts.idFile.value).lines("UTF-8").toSet[String] else Set[String]()
    val pred = new ArrayBuffer[(String,String)](1000000)
    val gold = new ArrayBuffer[(String,String)](1000000)
    new File(opts.predicted.value).lines("UTF-8").foreach{
      line =>
        val splt = line.split("\t")
        if (ids.isEmpty || ids.contains(splt(0))) {
          pred += ((splt(0),splt(1)))
        }
    }
    new File(opts.gold.value).lines("UTF-8").foreach{
      line =>
        val splt = line.split("\t")
        if (ids.isEmpty || ids.contains(splt(0))) {
          gold += ((splt(0),splt(1)))
        }
    }
    assert(pred.size == gold.size)
    val pw = Pairwise.apply(new BasicEvaluatableClustering(pred), new BasicEvaluatableClustering(gold))
    println(s"${opts.algorithm.value}\t${opts.dataset.value}\t${pw.precision}\t${pw.recall}\t${pw.f1}")
  }


}