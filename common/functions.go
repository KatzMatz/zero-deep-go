package common

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func StepFunction(x float64) float64 {
	if x > 0.0 {
		return 1.0
	} else {
		return 0.0
	}
}

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func Relu(x float64) float64 {
	return math.Max(0, x)
}

func SoftMax(x mat.Matrix) mat.Matrix {
	row, _ := x.Dims()
	if row != 1 {
		panic("input matrix must be a vector")
	}

	c := maxAtRow(x, 0)
	a := SubConstant(x, c)
	exps := ApplyFunction(a, math.Exp)
	sumExps := SumRow(exps, 0)

	result := DivConstant(exps, sumExps)

	return result
}
