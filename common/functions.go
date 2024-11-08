package common

import (
	"math"
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
