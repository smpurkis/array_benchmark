import times
import sequtils


proc compute_array(m: int, n: int) =
  var arr = newSeqWith(m, newSeq[int](n))
  for i in 0..m-1:
    for j in 0..n-1:
      arr[i][j] = i*i + j*j  
  echo(arr[m-1][n-1])

var m = 15000
var n = 15000
var n_loops = 5
let time = cpuTime()
for i in 0..n_loops-1:
  compute_array(m, n)
echo(cpuTime() - time)