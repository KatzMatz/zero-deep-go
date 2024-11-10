package dataset

import (
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"

	"golang.org/x/sync/errgroup"
	"gonum.org/v1/gonum/mat"
)

var BASE_URL string = "https://storage.googleapis.com/cvdf-datasets/mnist/"
var FILE_NAMES []string = []string{
	"train-images-idx3-ubyte.gz",
	"train-labels-idx1-ubyte.gz",
	"t10k-images-idx3-ubyte.gz",
	"t10k-labels-idx1-ubyte.gz",
}

func DownLoadMnist() error {

	var eg errgroup.Group

	for _, f := range FILE_NAMES {

		_, err := os.Stat(f)
		if err == nil {
			fmt.Printf("[%s] is already exist. Skip Donwload.\n", f)
			continue
		}

		eg.Go(func() error {
			return downloadFile(BASE_URL, f)
		})
	}

	if err := eg.Wait(); err != nil {
		return err
	}

	return nil
}

const (
	IMAGE_ROW        int = 28
	IMAGE_COL        int = 28
	NUM_TRAIN_IMAGES int = 60000
	NUM_TEST_IMAGES  int = 10000

	MAX_PIXEL_VALUE int = 255
)

type Image = []uint8
type Label = uint8

type Mnist struct {
	TrainImage []Image
	TrainLabel []Label
	TestImage  []Image
	TestLabel  []Label
}

type NormalizedImage = []float64

type NormalizedMnist struct {
	TrainImage []NormalizedImage
	TrainLabel []Label
	TestImage  []NormalizedImage
	TestLabel  []Label
}

func (m *Mnist) Normalize() NormalizedMnist {

	normalizedTestImages := []NormalizedImage{}
	for idx := range NUM_TEST_IMAGES {
		normalizedTestImages = append(normalizedTestImages, normalizeImage(m.TestImage[idx]))
	}

	normalizedTrainImages := []NormalizedImage{}
	for idx := range NUM_TRAIN_IMAGES {
		normalizedTrainImages = append(normalizedTrainImages, normalizeImage(m.TrainImage[idx]))
	}

	return NormalizedMnist{
		TrainImage: normalizedTrainImages,
		TrainLabel: m.TrainLabel,
		TestImage:  normalizedTestImages,
		TestLabel:  m.TestLabel,
	}
}

func normalizeImage(image Image) NormalizedImage {
	normalized := NormalizedImage{}
	for _, v := range image {
		normalized = append(normalized, float64(v)/float64(MAX_PIXEL_VALUE))
	}
	return normalized
}

func LoadMnist() (Mnist, error) {
	if err := DownLoadMnist(); err != nil {
		return Mnist{}, err
	}

	trainImageBytes, err := loadPayLoad(FILE_NAMES[0])
	if err != nil {
		return Mnist{}, nil
	}
	trainLabel, err := loadPayLoad(FILE_NAMES[1])
	if err != nil {
		return Mnist{}, nil
	}
	testImagebytes, err := loadPayLoad(FILE_NAMES[2])
	if err != nil {
		return Mnist{}, nil
	}
	testLabel, err := loadPayLoad(FILE_NAMES[3])
	if err != nil {
		return Mnist{}, nil
	}

	trainImages := splitImage(trainImageBytes[16:], NUM_TRAIN_IMAGES)
	testImages := splitImage(testImagebytes[16:], NUM_TEST_IMAGES)

	return Mnist{
		TrainImage: trainImages,
		TrainLabel: trainLabel[8:],
		TestImage:  testImages,
		TestLabel:  testLabel[8:],
	}, nil
}

func splitImage(payload []byte, n int) []Image {
	offset := 0
	images := []Image{}

	imageSize := IMAGE_ROW * IMAGE_COL

	for range n {
		images = append(images, payload[offset:(offset+imageSize)])
		offset += imageSize
	}

	return images
}

func loadPayLoad(fileName string) ([]byte, error) {

	f, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}

	gzipReader, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}

	bytes, err := io.ReadAll(gzipReader)
	if err != nil {
		return nil, err
	}

	return bytes, nil
}

func downloadFile(BASE_URL, fileName string) error {
	requestUrl, err := url.JoinPath(BASE_URL, fileName)
	if err != nil {
		return err
	}

	fmt.Printf("Request to [%s] ...\n", requestUrl)
	response, err := http.Get(requestUrl)
	if err != nil {
		return err
	}

	bytes, err := io.ReadAll(response.Body)
	if err != nil {
		return err
	}

	outputFile, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer outputFile.Close()

	_, err = outputFile.Write(bytes)
	if err != nil {
		return err
	}

	fmt.Printf("Output to %s\n", fileName)

	return nil
}

func Image2Matrix(image Image) mat.Matrix {
	floats := []float64{}

	for _, v := range image {
		floats = append(floats, float64(v))
	}

	m := mat.NewDense(1, len(image), floats)

	return m
}

func NormalizedImage2Matrix(image NormalizedImage) mat.Matrix {
	floats := []float64{}

	for _, v := range image {
		floats = append(floats, float64(v))
	}

	m := mat.NewDense(1, len(image), floats)

	return m
}

func NormalizedImage2BatchMatrix(images []NormalizedImage, batchSize int) mat.Matrix {
	floats := []float64{}

	for idx := range batchSize {
		for _, v := range images[idx] {
			floats = append(floats, float64(v))
		}
	}

	m := mat.NewDense(batchSize, len(images[0]), floats)

	return m
}
