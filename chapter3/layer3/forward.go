package main

import (
	"fmt"

	"githhub.com/KatzMatz/zero-deep-go/common"
	"gonum.org/v1/gonum/mat"
)

func initNetwork() map[string]mat.Matrix {
	network := map[string]mat.Matrix{}

	network["W1"] = mat.NewDense(2, 3, []float64{0.1, 0.3, 0.5, 0.2, 0.4, 0.5})
	network["b1"] = mat.NewDense(1, 3, []float64{0.1, 0.2, 0.3})
	network["W2"] = mat.NewDense(3, 2, []float64{0.1, 0.4, 0.2, 0.5, 0.3, 0.6})
	network["b2"] = mat.NewDense(1, 2, []float64{0.1, 0.2})
	network["W3"] = mat.NewDense(2, 2, []float64{0.1, 0.3, 0.2, 0.4})
	network["b3"] = mat.NewDense(1, 2, []float64{0.1, 0.2})

	return network
}

func IdentifyFunc(x mat.Matrix) mat.Matrix {
	return x
}

func forward(network map[string]mat.Matrix, x mat.Matrix) mat.Matrix {
	w1 := network["W1"]
	w2 := network["W2"]
	w3 := network["W3"]
	b1 := network["b1"]
	b2 := network["b2"]
	b3 := network["b3"]

	a1 := common.Add(common.Dot(x, w1), b1)
	z1 := common.ApplyFunction(a1, common.Sigmoid)

	fmt.Printf("z1: %n\n", z1)

	a2 := common.Add(common.Dot(z1, w2), b2)
	z2 := common.ApplyFunction(a2, common.Sigmoid)

	a3 := common.Add(common.Dot(z2, w3), b3)
	y := IdentifyFunc(a3)

	return y
}

func main() {
	network := initNetwork()
	x := mat.NewDense(1, 2, []float64{1.0, 0.5})
	y := forward(network, x)

	fmt.Println(y)
}
