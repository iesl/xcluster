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

import java.util.concurrent.atomic.AtomicInteger


/**
  * Threadsafe way to increment counters.
  */
trait OperationCounter {

  var _on = false

  def on() = _on = true
  def off() = _on = false

  val name: String

  var count = new AtomicInteger(0)

  def increment() = if (_on) {
    count.incrementAndGet()
  }

}

object ComparisonCounter extends OperationCounter {
  override val name: String = "ComparisonCounter"
}