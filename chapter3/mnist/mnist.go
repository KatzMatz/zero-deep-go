package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"

	"githhub.com/KatzMatz/zero-deep-go/common"
	"githhub.com/KatzMatz/zero-deep-go/dataset"
	"gonum.org/v1/gonum/mat"
)

type SampleWeight struct {
	W1 [][]float64 `json:"W1"`
	W2 [][]float64 `json:"W2"`
	W3 [][]float64 `json:"W3"`
	B1 []float64   `json:"b1"`
	B2 []float64   `json:"b2"`
	B3 []float64   `json:"b3"`
}

func loadWeights(fileName string) (SampleWeight, error) {
	f, err := os.Open(fileName)
	if err != nil {
		return SampleWeight{}, nil
	}

	jsonData, err := io.ReadAll(f)
	if err != nil {
		return SampleWeight{}, nil
	}

	var weights SampleWeight
	err = json.Unmarshal(jsonData, &weights)

	if err != nil {
		return SampleWeight{}, nil
	}

	return weights, nil
}

func main() {

	fileName := "sample_weight.json"
	weights, err := loadWeights(fileName)
	if err != nil {
		panic(err)
	}

	w1 := common.Slice2Matrix(weights.W1)
	w2 := common.Slice2Matrix(weights.W2)
	w3 := common.Slice2Matrix(weights.W3)
	b1 := mat.NewDense(1, len(weights.B1), weights.B1)
	b2 := mat.NewDense(1, len(weights.B2), weights.B2)
	b3 := mat.NewDense(1, len(weights.B3), weights.B3)
	mnist, err := dataset.LoadMnist()
	normalizedMnist := mnist.Normalize()
	accuracy := 0.0

	for idx := range dataset.NUM_TEST_IMAGES {
		x := dataset.NormalizedImage2Matrix(normalizedMnist.TestImage[idx])

		a1 := common.Add(common.Dot(x, w1), b1)
		z1 := common.ApplyFunction(a1, common.Sigmoid)

		a2 := common.Add(common.Dot(z1, w2), b2)
		z2 := common.ApplyFunction(a2, common.Sigmoid)

		a3 := common.Add(common.Dot(z2, w3), b3)
		y := common.SoftMax(a3)
		predict := common.ArgMax(y)

		// (Answer, Predict) =
		fmt.Printf("(%d, %d), ", normalizedMnist.TestLabel[idx], predict)

		if normalizedMnist.TestLabel[idx] == uint8(predict) {
			accuracy += 1.0
		}

	}
	fmt.Println("")

	fmt.Println(accuracy / float64(dataset.NUM_TEST_IMAGES))
}
