package main

import (
	"flag"
	"fmt"
	"github.com/davecgh/go-spew/spew"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
	"gocv.io/x/gocv"
	"image"
	"io/ioutil"
	"math"
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

func resize_image(uri string) (gocv.Mat, float64, float64) {
	max_side_len := 2400
	im := gocv.IMRead(uri, gocv.IMReadColor)

	dims := im.Size()

	h := dims[0]
	w := dims[1]

	resize_h := h
	resize_w := w
	ratio := 0.0

	if math.Max(float64(resize_h), float64(resize_w)) > float64(max_side_len) {
		if resize_h > resize_w {
			ratio = float64(max_side_len) / float64(resize_h)
		} else {
			ratio = float64(max_side_len) / float64(resize_w)
		}
	} else {
		ratio = 1.
	}

	resize_h = int(float64(resize_h) * ratio)
	resize_w = int(float64(resize_w) * ratio)

	if resize_h%32 == 0 {
		resize_h = resize_h
	} else {
		resize_h = (resize_h / 32) * 32
	}

	if resize_w%32 == 0 {
		resize_w = resize_w
	} else {
		resize_w = (resize_w / 32) * 32
	}

	size := image.Point{X: resize_w, Y: resize_h}

	gocv.Resize(im, &im, size, 0, 0, gocv.InterpolationLinear)

	ratio_h := float64(resize_h) / float64(h)
	ratio_w := float64(resize_w) / float64(w)

	return im, ratio_h, ratio_w
}

func main() {
	img_resized, ratio_h, ratio_w := resize_image("test.png")
	_ = ratio_w
	_ = ratio_h

	img_resized.DivideFloat(127.5)
	img_resized.SubtractFloat(1)

	bytes, err := gocv.IMEncode(gocv.PNGFileExt, img_resized)

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

	image, err := tf.NewTensor(string(bytes))
	if err != nil {
		fmt.Println(err)
		return
	}

	tensor, err := imageResizer.Run(image)
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
