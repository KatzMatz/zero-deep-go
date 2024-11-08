package common

import (
	"gonum.org/v1/gonum/mat"
)

// m x n . n x l -> m x l
func Dot(a, b mat.Matrix) mat.Matrix {
	row, _ := a.Dims()
	_, col := b.Dims()

	c := mat.NewDense(row, col, nil)
	c.Mul(a, b)

	return c
}

func Add(a, b mat.Matrix) mat.Matrix {
	row, col := a.Dims()
	c := mat.NewDense(row, col, nil)
	c.Add(a, b)

	return c
}

func SubConstant(a mat.Matrix, v float64) mat.Matrix {
	row, col := a.Dims()
	constants := mat.NewDense(row, col, RepeatSlice(v, (row*col)))

	b := mat.NewDense(row, col, nil)
	b.Sub(a, constants)

	return b
}

func DivConstant(x mat.Matrix, v float64) mat.Matrix {
	row, col := x.Dims()
	constants := mat.NewDense(row, col, RepeatSlice(v, (row*col)))

	result := mat.NewDense(row, col, nil)
	result.DivElem(x, constants)

	return result
}

func RepeatSlice[T int | float32 | float64](v T, n int) []T {
	s := []T{}

	for range n {
		s = append(s, v)
	}

	return s
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

func maxAtRow(x mat.Matrix, row int) float64 {
	_, c := x.Dims()

	result := x.At(row, 0)
	for idx := range c {
		if result < x.At(row, idx) {
			result = x.At(row, idx)
		}
	}

	return result
}

func SumRow(x mat.Matrix, row int) float64 {
	_, col := x.Dims()

	sum := 0.0
	for idx := range col {
		sum += x.At(row, idx)
	}

	return sum
}
