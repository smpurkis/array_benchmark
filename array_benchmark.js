// const tf = require('@tensorflow/tfjs-node');
const ma = require("mathjs")

function compute_tf(m, n) {
    x = tf.broadcastTo(tf.pow(tf.reshape(tf.range(1, m+1), [-1, 1]), 2), [m, n]) + tf.broadcastTo(tf.pow(tf.range(1, n+1), 2), [m, n])
    return x
}

function compute_mathjs_range(m, n) {
    a = ma.square(ma.range(0, m))
    a = ma.matrix(new Array(n).fill(a))
    b = ma.square(ma.range(0, n))
    b = ma.matrix(new Array(m).fill(b))
    b = ma.transpose(b)
    x = ma.add(a, b)
    // console.log(x.get([m-1, n-1]))
    return x
}


function compute_range_array(m, n) {
    let arr = ma.zeros(m, n);
    for (var i = 0; i < m; i++) {
        for (var j = 0; j < n; j++) {
            arr.set([i, j], i*i + j*j);
        }
    }
    // console.log(arr[m-1][n-1]);
    return arr
}

function compute_range_lookup(m, n) {
    let arr = [];
    for (var i = 0; i < m; i++) {
        arr_i = []
        for (var j = 0; j < n; j++) {
            arr_i[j] = i*i + j*j;
        }
        arr[i] = arr_i;
    }
    // console.log(arr[m-1][n-1]);
    return arr
}

function compute_range(m, n) {
    let arr = [];
    for (var i = 0; i < m; i++) {
        arr[i] = [];
        for (var j = 0; j < n; j++) {
            arr[i][j] = i*i + j*j;
        }
    }
    // console.log(arr[m-1][n-1]);
    return arr
}


m = 10000
n = 10000
n_loop = 1
// s = Date.now()
// for (i=0; i<10; i++) {
//     compute_range(m, n)
// }
// console.log(`${Date.now() - s}`);

s = Date.now()
for (i=0; i<10; i++) {
    compute_range_lookup(m, n)
}
console.log(`${Date.now() - s}`);

// s = Date.now()
// for (i=0; i<10; i++) {
//     compute_range_array(m, n)
// }
// console.log(`${Date.now() - s}`);

// s = Date.now()
// for (i=0; i<10; i++) {
//     compute_mathjs_range(m, n)
// }
// console.log(`${Date.now() - s}`);