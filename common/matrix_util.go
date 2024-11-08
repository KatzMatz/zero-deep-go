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
