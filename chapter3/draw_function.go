package main

import (
	"fmt"

	"githhub.com/KatzMatz/zero-deep-go/common"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func drawFunction(start, stop, step float64, f func(x float64) float64, outputFilePath string) error {
	x := common.ARangeVector(start, stop, step)
	y := common.ApplyFunction(x, f)

	plot := plot.New()
	plot.X.Min = -1.0 * (start + 1.0)
	plot.X.Max = (stop + 1.0)

	col, _ := x.Dims()
	points := make(plotter.XYs, col)
	for idx := range points {
		points[idx].X = x.At(idx, 0)
		points[idx].Y = y.At(idx, 0)
	}
	line, err := plotter.NewLine(points)
	if err != nil {
		panic(err)
	}
	plot.Add(line)

	err = plot.Save(4*vg.Inch, 4*vg.Inch, outputFilePath)

	return err
}

func main() {

	err := drawFunction(-5.0, 5.0, 0.1, common.StepFunction, "step_func.png")
	if err != nil {
		fmt.Printf("failed plot graph: %e\n", err)
	}

	err = drawFunction(-5.0, 5.0, 0.1, common.Sigmoid, "sigmoid.png")
	if err != nil {
		fmt.Printf("failed plot graph: %e\n", err)
	}
}
