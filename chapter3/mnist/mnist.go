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

func initNetwork(weights SampleWeight) map[string]mat.Matrix {

	network := map[string]mat.Matrix{}

	network["w1"] = common.Slice2Matrix(weights.W1)
	network["w2"] = common.Slice2Matrix(weights.W2)
	network["w3"] = common.Slice2Matrix(weights.W3)
	network["b1"] = mat.NewDense(1, len(weights.B1), weights.B1)
	network["b2"] = mat.NewDense(1, len(weights.B2), weights.B2)
	network["b3"] = mat.NewDense(1, len(weights.B3), weights.B3)

	return network
}

func predict(network map[string]mat.Matrix, x mat.Matrix) mat.Matrix {

	w1 := network["w1"]
	w2 := network["w2"]
	w3 := network["w3"]
	b1 := network["b1"]
	b2 := network["b2"]
	b3 := network["b3"]

	a1 := common.AddBroadCast(common.Dot(x, w1), b1)
	z1 := common.ApplyFunction(a1, common.Sigmoid)

	a2 := common.AddBroadCast(common.Dot(z1, w2), b2)
	z2 := common.ApplyFunction(a2, common.Sigmoid)

	a3 := common.AddBroadCast(common.Dot(z2, w3), b3)
	y := common.SoftMaxMatrix(a3)

	return y
}

func main() {

	fileName := "sample_weight.json"
	weights, err := loadWeights(fileName)
	if err != nil {
		panic(err)
	}
	network := initNetwork(weights)

	mnist, err := dataset.LoadMnist()
	normalizedMnist := mnist.Normalize()
	accuracy := 0.0

	for idx := range dataset.NUM_TEST_IMAGES {
		x := dataset.NormalizedImage2Matrix(normalizedMnist.TestImage[idx])
		y := predict(network, x)
		predict := common.ArgMaxEachRow(y)

		// (Answer, Predict) =
		fmt.Printf("(%d, %d), ", normalizedMnist.TestLabel[idx], predict[0])

		if normalizedMnist.TestLabel[idx] == uint8(predict[0]) {
			accuracy += 1.0
		}

	}
	fmt.Println("")

	fmt.Println(accuracy / float64(dataset.NUM_TEST_IMAGES))

	// predict by batch
	batchSize := 100
	countCorrect := 0
	for idx := 0; idx < dataset.NUM_TEST_IMAGES; idx += batchSize {
		xs := dataset.NormalizedImage2BatchMatrix(normalizedMnist.TestImage[idx:(idx+batchSize)], batchSize)
		y := predict(network, xs)

		predicts := common.ArgMaxEachRow(y)
		countCorrect += CountBatchCorrectPredicts(predicts, normalizedMnist.TestLabel[idx:(idx+batchSize)], batchSize)
	}

	fmt.Println("Batch accuracy")
	fmt.Printf("%f\n", float64(countCorrect)/float64(dataset.NUM_TEST_IMAGES))

}

func CountBatchCorrectPredicts(y []int, label []uint8, batchSize int) int {
	if len(y) != len(label) {
		panic("length is not correct")
	}
	count := 0

	for idx := range batchSize {
		if y[idx] == int(label[idx]) {
			count += 1
		}
	}

	return count
}
