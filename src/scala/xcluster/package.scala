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

import java.io.{FileInputStream, InputStreamReader, BufferedReader, File}
import java.io._
import java.util.UUID

import cc.factorie.util.Threading

import scala.collection.mutable.ArrayBuffer
import cc.factorie._

package object xcluster {

  implicit class XClusterIndexedSeqExtras[T](seq: IndexedSeq[T]) {
    def pairsParallel(numThreads: Int): Iterable[(T,T)] = {
      val initSize = if (seq.size * (seq.size - 1) > 0) seq.size * (seq.size - 1) else Int.MaxValue-1000000
      val pairs = new ArrayBuffer[(T,T)](initSize)
//      println(s"[IndexedSeqExtras] ${seq.size * (seq.size - 1)} Pairs to find")
      Threading.parForeach(seq.indices,numThreads)(idx => {
        var i = idx + 1
        val p = new ArrayBuffer[(T,T)](seq.length - i)
        while (i < seq.length) {
          p += ((seq(idx),seq(i)))
          i += 1
        }
        synchronized {
          pairs ++= p
//          print(s"\r[IndexedSeqExtras] ${pairs.size} pairs found")
        }
      })
//      println(s"\n[IndexedSeqExtras] Done finding pairs")
      pairs
    }
  }

  implicit class XClusterArrayDoubleExtras(arr: Array[Double]) {

    def += (arr2: Array[Double]): Unit = {
      val len = arr.length
      assert(arr2.length == len)
      var i = 0
      while (i < len) {
        arr(i) += arr2(i)
        i += 1
      }
    }

    def / (n: Double) = {
      val len = arr.length
      val newArr = new Array[Double](len)
      var i = 0
      while (i < len) {
        newArr(i) = arr(i) / n
        i += 1
      }
      newArr
    }
  }

  implicit class XClusterFileExtras(file: File) {
    def lines(codec: String) = {
      new BufferedReader(new InputStreamReader(new FileInputStream(file),codec)).toIterator
    }

    def allFilesRecursively(filter: File => Boolean = _ => true): Iterable[File] = {
      if (file.isDirectory)
        file.listFiles().toIterable.filterNot(_.isDirectory).filter(filter) ++
          file.listFiles().toIterable.filter(_.isDirectory).map( f => f.allFilesRecursively(filter)).flatten
      else
      if (filter(file))
        Some(file)
      else
        None
    }


  }

  implicit class XClusterIterableDoubleExtras(i: Iterable[Double]) {

    def average = {
      var s = 0.0
      var c = 0
      i.foreach{
        ii =>
          s += ii
          c += 1
      }
      s / c
    }

    def fastSum(fn: Double => Double) = {
      var s = 0.0
      i.foreach{
        ii =>
          s += fn(ii)
      }
      s
    }

    def variance = {
      val avg = i.average
      var v = 0.0
      var c = 0
      i.foreach{
        ii =>
          val diff = ii - avg
          v += diff * diff
          c  += 1
      }
      v / c
    }

  }

  def randomId: String = UUID.randomUUID().toString



  implicit class XClusterStringExtras(str: String) {

    def toFile(file: File,codec: String = "UTF-8"): Unit = {
      val pw = new PrintWriter(file,codec)
      pw.print(str)
      pw.close()
    }
  }

}
