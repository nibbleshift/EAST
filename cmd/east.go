package main

import (
	"bytes"
	"flag"
	"fmt"
	"github.com/davecgh/go-spew/spew"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"image"
	"io/ioutil"
	"os"

	"github.com/nfnt/resize"
	"image/png"
)

var (
	modelPath = flag.String("model_path", "", "Model path to load")
	imagePath = flag.String("image_path", "", "Image path to evaluate")
)

func init() {
	flag.Parse()
}

// ImageResizer holds a graph to resize and normalize an image
type ImageResizer struct {
	Graph   *tf.Graph
	Session *tf.Session
	Input   tf.Output
	Output  tf.Output
}

// NewImageResizer returns an ImageResizer instance that decode an image and resize to a given shape
// TODO: maybe better to use this params as placeholders in the graph
// Note that building the graph adds a lot of overhead, so it's good practice to build it only once
func NewImageResizer(width, height, channels int) (*ImageResizer, error) {
	var (
		graph         *tf.Graph
		input, output tf.Output
	)
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ResizeBilinear(s,
		op.ExpandDims(s,
			op.Cast(s,
				op.DecodePng(s, input, op.DecodePngChannels(int64(channels))), tf.Float),
			op.Const(s.SubScope("make_batch"), int32(0))),
		op.Const(s.SubScope("size"), []int32{int32(height), int32(width)}))
	graph, err := s.Finalize()
	if err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	return &ImageResizer{
		Graph:   graph,
		Session: session,
		Input:   input,
		Output:  output}, nil
}

// Run reads a PNG-encoded string tensor and returns a tensor with the resized shape 1 x height x width x channels.
func (I *ImageResizer) Run(tensor *tf.Tensor) (*tf.Tensor, error) {
	normalized, err := I.Session.Run(
		map[tf.Output]*tf.Tensor{I.Input: tensor},
		[]tf.Output{I.Output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

func resizeImage(img image.Image) (resized image.Image, ratioH float64, ratioW float64) {
	h := img.Bounds().Dy()
	w := img.Bounds().Dx()

	// Limit the max size of the image
	maxSize := 2400

	// Resize ratio
	ratio := 1.0

	biggestDim := w
	if h > w {
		biggestDim = h
	}

	if biggestDim > maxSize {
		ratio = float64(maxSize / biggestDim)
	}

	resizeH := uint(float64(h) * ratio)
	resizeW := uint(float64(w) * ratio)

	if resizeH%32 != 0 {
		resizeH = (resizeH / 32.0) * 32
	}

	if resizeW%32 != 0 {
		resizeW = (resizeW / 32) * 32
	}

	resized = resize.Resize(resizeW, resizeH, img, resize.Bilinear)

	ratioH = float64(resizeH) / float64(h)
	ratioW = float64(resizeW) / float64(w)
	return resized, ratioH, ratioW
}

func prepareImage(path string, resizer *ImageResizer) (*tf.Tensor, error) {
	reader, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	img, err := png.Decode(reader)
	if err != nil {
		return nil, err
	}

	img, _, _ = resizeImage(img)

	// TODO
	// img_resized.DivideFloat(127.5)
	// img_resized.SubtractFloat(1)

	var buf bytes.Buffer
	err = png.Encode(&buf, img)
	if err != nil {
		return nil, err
	}

	tensor, err := tf.NewTensor(string(buf.Bytes()))
	if err != nil {
		return nil, err
	}

	return resizer.Run(tensor)
}

func main() {
	model, err := ioutil.ReadFile(*modelPath)
	if err != nil {
		fmt.Println(err)
		return
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		fmt.Println(err)
		return
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	defer session.Close()

	imageResizer, err := NewImageResizer(512, 512, 3)
	if err != nil {
		fmt.Println(err)
		return
	}

	tensor, err := prepareImage(*imagePath, imageResizer)
	if err != nil {
		fmt.Println(err)
		return
	}

	input := graph.Operation("input_image").Output(0)

	score_out := graph.Operation("pred_score_map/Sigmoid").Output(0)
	geo_out := graph.Operation("pred_geo_map/concat").Output(0)

	spew.Dump(score_out)
	spew.Dump(geo_out)

	result, err := session.Run(
		map[tf.Output]*tf.Tensor{
			input: tensor,
		},
		[]tf.Output{
			score_out,
			geo_out,
		},
		nil,
	)

	spew.Dump(result)
}
