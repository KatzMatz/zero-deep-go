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

func ApplyFunction(x mat.Matrix, f func(v float64) float64) mat.Matrix {
	var y mat.Dense
	y.Apply(func(_, _ int, v float64) float64 {
		return f(v)
	}, x)
	return &y
}

func ARange(start, stop, step float64) []float64 {
	result := []float64{}
	for v := start; v <= stop; v += step {
		result = append(result, v)
	}

	return result
}

func ARangeVector(start, stop, step float64) mat.Matrix {
	s := ARange(start, stop, step)
	return mat.NewVecDense(len(s), s)
}
