package common

import "gonum.org/v1/gonum/mat"

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
