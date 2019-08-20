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

package xcluster.utils

import java.io.PrintWriter
import java.util

import xcluster.eval.{EvalTreeNode, LoadEvalTree}

import scala.collection.JavaConverters._

class Graphviz {

  val internal_color = "lavenderblush4"
  val colors = IndexedSeq("aquamarine","bisque","blue","blueviolet",
  "brown","cadetblue","chartreuse","coral","cornflowerblue",
  "crimson","darkgoldenrod","darkgreen",
  "darkkhaki","darkmagenta","darkorange",
  "darkred","darksalmon","darkseagreen","darkslateblue","darkslategrey",
  "darkviolet","deepskyblue","dodgerblue","firebrick",
  "forestgreen","gainsboro","ghostwhite","gold","goldenrod","gray","grey","green",
  "greenyellow","honeydew","hotpink","indianred","indigo","ivory","khaki",
  "lavender","lavenderblush","lawngreen","lemonchiffon","lightblue",
  "lightcoral","lightcyan","lightgoldenrodyellow","lightgray","lightgreen",
  "lightgrey","lightpink","lightsalmon","lightseagreen","lightskyblue","lightslategray",
  "lightslategrey","lightsteelblue","lightyellow","limegreen","linen","magenta","maroon",
  "mediumaquamarine","mediumblue","mediumorchid","mediumpurple","mediumseagreen",
  "mediumslateblue","mediumturquoise","midnightblue","mintcream","mistyrose",
  "moccasin","navajowhite","navy","oldlace","olive","olivedrab","orange",
  "orangered","orchid","palegoldenrod","palegreen","paleturquoise","palevioletred",
  "papayawhip","peachpuff","peru","pink","powderblue","purple","red","rosybrown",
  "royalblue","saddlebrown","salmon","sandybrown","seagreen","seashell","sienna",
  "silver","skyblue","slateblue","slategray","slategrey","snow","springgreen",
  "steelblue","tan","teal","thistle","tomato","violet","wheat","burlywood","chocolate")
  var color_map = new util.HashMap[String,String]().asScala
  var color_counter = 0

  def formatId(id: String) = "id" + clean_label(id.replaceAll("-",""))

  def get_node_label(node: EvalTreeNode) = {
    val lbl = new StringBuilder()
    lbl append formatId(node.id)
    lbl append "<BR/>"
    lbl append s"num pts: ${node.leaves().size}"
    lbl append "<BR/>"
    try {
      lbl append s"purity: ${node.purity()}"
      lbl append "<BR/>"
    } catch {
      case _: Exception => {}
    }
    val cc = node.labels.groupBy(identity).mapValues(_.size).toIndexedSeq.sortBy(-_._2).take(10)
    lbl append s"classes: ${cc.map(c => clean_label(c._1) + "x" + c._2).mkString(" ")}"
    lbl.toString()
  }

  def clean_label(s: String) = {
    s.replaceAll("[^a-zA-Z0-9\\-]", "_").replaceAll("[/:.]","_")
  }

  def _get_color(lbl: String) = {
    if (this.color_map.contains(lbl))
      this.color_map(lbl)
    else {
      this.color_map.put(lbl,this.colors(this.color_counter))
      this.color_counter = (this.color_counter + 1) % this.colors.size
      this.color_map(lbl)
    }
  }

  def _format_graphviz_node(node: EvalTreeNode) = {
    val sb = new StringBuilder()
    var color = this.internal_color
    try {
      if (node.purity() < 1.0)
        color = this.internal_color
      else
        color = this._get_color(node.labels.head)
    } catch {
      case _: Exception => {}
    }
//    val color = if (node.purity() < 1.0) this.internal_color else this._get_color(node.labels.head)
    val shape = "egg"
    if (node.parent.isEmpty) {
      sb append s"\n${formatId(node.id)}[shape=${shape};style=filled;color=${color};label=<${get_node_label(node)}<BR/>${color}<BR/>>]"
      sb append "\nROOTNODE[shape=star;style=filled;color=gold;label=<ROOT>]"
      sb append s"\nROOTNODE->${formatId(node.id)}"
    } else {
      sb append s"\n${formatId(node.id)}[shape=${shape};style=filled;color=${color};label=<${get_node_label(node)}<BR/>${color}<BR/>>]"
      sb append s"\n${formatId(node.parent.get.id)}->${formatId(node.id)}"
    }
    sb.toString()
  }


  def graphvis_tree(root: EvalTreeNode) = {
    val sb = new StringBuilder()
    sb append "digraph TreeStructure {\n"
    sb append this._format_graphviz_node(root)
    root.descendants().foreach {
      d =>
        sb append _format_graphviz_node(d)
    }
    sb append "\n}"
    sb.toString()
  }

}

object Graphviz {

  def toGV(root:EvalTreeNode) = {
    val gv = new Graphviz
    gv.graphvis_tree(root)
  }

  def main (args: Array[String]) {
    val pw = new PrintWriter(args(1),"UTF-8")
    val tree = LoadEvalTree.load(args(0))
    pw.println(toGV(tree))
    pw.close()
  }

}
