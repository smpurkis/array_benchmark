package main

import (
	"fmt"
	"time"
)

func main() {
	m := 15000
	n := 15000
	n_loop := 5
	start := time.Now()

	for i := 0; i < n_loop; i++ {
		compute_range(m, n)
	}

	duration := time.Since(start)
	fmt.Println(duration)
}

func compute_range(m2 int, n2 int) {
	arr := [15000][15000]int{}
	for j := 0; j < m2; j++ {
		for k := 0; k < n2; k++ {
			arr[j][k] = j*j + k*k
		}
	}
	fmt.Println(arr[15000-1][15000-1])
}
