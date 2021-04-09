import times
import sequtils


proc compute_array(m: any, n: any) =
  var arr = newSeqWith(m, newSeq[int](n))
  for i in 0..m-1:
    for j in 0..n-1:
      arr[i][j] = i*i + j*j  
  echo(arr[m-1][n-1])

var m = 10000
var n = 10000
var n_loops = 5
let time = now()
for i in 0..n_loops-1:
  compute_array(m, n)
echo(now() - time)
