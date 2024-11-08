package main

import (
	"fmt"

	"githhub.com/KatzMatz/zero-deep-go/common"
	"gonum.org/v1/gonum/mat"
)

func main() {

	a := mat.NewDense(1, 3, []float64{0.3, 2.9, 4.0})
	y := common.SoftMax(a)

	fmt.Println(y)
}
