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
	var m mat.Dense
	m.Add(a, b)

	return &m
}

func AddBroadCast(m, vec mat.Matrix) mat.Matrix {

	vector := GetRow(vec, 0)

	row, col := m.Dims()
	d := mat.NewDense(row, col, nil)
	for idx := range row {
		d.SetRow(idx, vector)
	}

	result := mat.NewDense(row, col, nil)
	result.Add(m, d)

	return result
}

func GetRow(m mat.Matrix, row int) []float64 {

	vec := []float64{}

	_, col := m.Dims()
	for idx := range col {
		vec = append(vec, m.At(row, idx))
	}

	return vec
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

func RepeatSlice[T any](v T, n int) []T {
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

func Slice2Matrix(x [][]float64) mat.Matrix {

	r := len(x)
	c := len(x[0])

	flatten := []float64{}

	for _, row := range x {
		flatten = append(flatten, row...)
	}
	matrix := mat.NewDense(r, c, flatten)

	return matrix
}

func SumRow(x mat.Matrix, row int) float64 {
	_, col := x.Dims()

	sum := 0.0
	for idx := range col {
		sum += x.At(row, idx)
	}

	return sum
}

func ArgMaxAtRow(x mat.Matrix, row int) int {
	_, col := x.Dims()

	argMax := 0
	maxValue := x.At(row, 0)
	for idx := range col {
		if x.At(row, idx) > maxValue {
			maxValue = x.At(row, idx)
			argMax = idx
		}
	}

	return argMax
}

func ArgMax(x mat.Matrix) int {
	row, col := x.Dims()

	argMaxRow, argMaxCol := 0, 0
	maxValue := x.At(0, 0)

	for r := range row {
		for c := range col {
			if x.At(r, c) > maxValue {
				maxValue = x.At(r, c)
				argMaxRow, argMaxCol = r, c
			}
		}
	}

	return (argMaxRow * col) + argMaxCol
}

func ArgMaxEachRow(x mat.Matrix) []int {
	row, col := x.Dims()

	result := []int{}
	for r := range row {
		maxCol, maxValue := 0, x.At(r, 0)
		for c := range col {
			if x.At(r, c) > maxValue {
				maxCol, maxValue = c, x.At(r, c)
			}
		}
		result = append(result, maxCol)
	}

	return result
}
