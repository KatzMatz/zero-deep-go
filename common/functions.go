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

func SoftMaxMatrix(x mat.Matrix) mat.Matrix {
	row, col := x.Dims()

	floats := []float64{}

	for idx := range row {
		c := maxAtRow(x, idx)
		expArgs := applyFunctionWithArg(GetRow(x, idx), sub, c)
		exps := applyFunction(expArgs, math.Exp)
		sumExps := fold(exps, func(a, b float64) float64 { return a + b }, 0)

		rowSoftMax := applyFunctionWithArg(exps, div, sumExps)
		floats = append(floats, rowSoftMax...)
	}

	result := mat.NewDense(row, col, floats)

	return result
}

func fold[T int | float64](s []T, f func(a, b T) T, initial T) T {
	result := initial
	for _, v := range s {
		result += f(result, v)
	}

	return result
}

func sub(a, b float64) float64 {
	return (a - b)
}

func div(a, b float64) float64 {
	return (a / b)
}

func applyFunction(s []float64, f func(v float64) float64) []float64 {
	result := []float64{}

	for _, value := range s {
		result = append(result, f(value))
	}

	return result
}

func applyFunctionWithArg(s []float64, f func(v, arg float64) float64, c float64) []float64 {
	result := []float64{}

	for _, value := range s {
		result = append(result, f(value, c))
	}

	return result
}
