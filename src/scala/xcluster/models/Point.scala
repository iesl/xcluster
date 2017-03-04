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

import xcluster._

/**
  * Point objects are in the input to clustering algorithms.
  * Datasets are made up of a collection of Points.
  * @param pid A unique id for the Point
  * @param label The ground truth cluster/class label of the point.
  * @param value The point itself, a vector of doubles.
  * @param weight (Optional) Associate a weight with the point. Default 1.0.
  */
case class Point(pid: String, label: String, var value: Array[Double],weight: Double = 1.0) {

  /**
    * The hash code of a point is based on its unique pid.
    * @return
    */
  override def hashCode(): Int = pid.hashCode


  /**
    * The point value divided by the weight.
    * Note that this is a lazy val not a def. And so after this is
    * constructed, subsequent changes to value will not effect this.
    */
  lazy val normalizedValue: Array[Double] = value / weight

}
