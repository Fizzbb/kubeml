package model

import (
	"fmt"
	"github.com/RedisAI/redisai-go/redisai"
	"github.com/diegostock12/thesis/ml/pkg/api"
	"github.com/gomodule/redigo/redis"
	"go.uber.org/zap"
	"gorgonia.org/tensor"
	"strconv"
	"sync"
)

type (

	// Holds the Layers of the model
	Model struct {
		logger *zap.Logger

		// Id of the parameter server
		jobId string

		Name string

		// StateDict holds the layer names
		// and the layers of the model. Each
		// layer has a bias and a weight
		StateDict map[string]*Layer

		// layerNames holds the names of the layers
		// which will be used to build the model for the
		// first time
		layerNames []string

		// lr must be float32 to be the same type as the tensors
		lr      float32
		lrSched LrScheduler

		redisClient *redisai.Client

		// Internal Lock to be applied during the update
		// TODO looks like each tensor has its own lock. If this is the case maybe we can speed things up
		mu sync.Mutex
	}

	// Layer keeps the Weights and Bias of a certain layer of the Neural Network
	Layer struct {
		Name string

		Weights *tensor.Dense

		HasBias bool
		Bias    *tensor.Dense
	}

	// Gradient saves the gradients of a layer
	Gradient struct {
		Weights *tensor.Dense

		HasBias bool
		Bias    *tensor.Dense
	}

	// Just a learning rate scheduler that multiplies the rate by rate when invoked
	LrScheduler struct {
		rate float32
	}
)

// Creates a new model with the specified layers
func NewModel(logger *zap.Logger, jobId string, task api.TrainRequest,
	layerNames []string, client *redisai.Client) *Model {
	return &Model{
		logger:      logger.Named("model"),
		Name:        task.ModelType,
		jobId:       jobId,
		layerNames:  layerNames,
		StateDict:   make(map[string]*Layer),
		lr:          task.LearningRate,
		redisClient: client,
	}
}

// Build gets all the initialized layers from the database
// Build should be called once just after the network is initialized by a worker
func (m *Model) Build() error {
	// For each layer name create a new layer with the tensors from the database
	m.logger.Debug("Building the model", zap.String("jobId", m.jobId))

	for _, layerName := range m.layerNames {

		m.logger.Debug("Creating new layer", zap.String("layerName", layerName))
		l, err := newLayer(m.logger, m.redisClient, layerName, m.jobId)
		if err != nil {
			m.logger.Error("Error building layer",
				zap.String("layer", layerName),
				zap.Error(err))
			return err
		}

		// Add it to the statedict
		m.StateDict[layerName] = l
	}

	return nil
}

// Update applies a set of gradients to all the layers
// Simply iterate through the model layers and update each with the gradients
// Simply use the layer names of the model with the -bias-grad added to them
// TODO seems like the layers already have a lock so maybe we do not need the mutex here
func (m *Model) Update(funcId int) error {

	m.logger.Info("Updating model...")

	// lock the model
	m.mu.Lock()
	defer m.mu.Unlock()

	for name, layer := range m.StateDict {

		// Get the gradients from the database
		g, err := newGradient(m.redisClient, name, m.jobId, funcId)
		if err != nil {
			m.logger.Error("Could not build gradient",
				zap.String("layer", name),
				zap.Error(err))
			return err
		}

		// update the layer
		err = layer.update(g, m.lr)
		if err != nil {
			m.logger.Error("Could not update layer",
				zap.String("layer", name),
				zap.Error(err))
			return err
		}

	}

	m.logger.Info("Updated model")
	return nil
}

// Summary runs through the layers of a model and prints its info
func (m *Model) Summary() {
	for name, layer := range m.StateDict {
		m.logger.Info("Layer",
			zap.String("name", name),
			zap.Any("shape", layer.Weights.Shape()),
			zap.Bool("bias", layer.HasBias),
		)
	}

}

// Save saves the new updated weights and bias in the database so it can be retrieved
// by the following functions
// TODO we could use pipeline to speed it up
func (m *Model) Save() error {
	m.logger.Info("Publishing model on the database")

	for name, layer := range m.StateDict {

		m.logger.Debug("Setting weights",
			zap.String("layer", name),
			zap.Any("shape", layer.Weights))
		args, _ := makeArgs(m.jobId, name, api.WeightSuffix, layer.Weights.Shape(), layer.Weights.Data())

		_, err := m.redisClient.DoOrSend("AI.TENSORSET", *args, nil)
		if err != nil {
			m.logger.Error("Error setting weights",
				zap.String("layer", name),
				zap.Error(err))
			return err
		}

		// Set the bias only if it is needed
		if layer.HasBias {
			m.logger.Debug("Setting bias",
				zap.String("layer", name),
				zap.Any("shape", layer.Bias))
			args, _ = makeArgs(m.jobId, name, api.BiasSuffix, layer.Bias.Shape(), layer.Bias.Data())

			_, err = m.redisClient.DoOrSend("AI.TENSORSET", *args, nil)
			if err != nil {
				m.logger.Error("Error setting bias",
					zap.String("layer", name),
					zap.Error(err))
				return err
			}
		}

	}

	m.logger.Info("Model published in the DB")
	return nil

}

// Build a new layer by getting it from the database already initialized
func newLayer(logger *zap.Logger, redisClient *redisai.Client, name, psId string) (*Layer, error) {

	// Get the redis keys
	weightName, biasName := getWeightKeys(name, false, psId, "")

	// Build the weight tensor
	logger.Debug("Loading the weights...")

	sWeights, weightValues, err := fetchTensor(redisClient, weightName)
	if err != nil {
		return nil, err
	}

	dimWeights := shapeToIntArray(sWeights...)
	w := tensor.New(tensor.WithShape(dimWeights...), tensor.WithBacking(weightValues))

	// If we have to build the bias tensor
	var b *tensor.Dense
	biasExists, err := tensorExists(redisClient, biasName)
	if err != nil {
		return nil, err
	}

	hasBias := true
	if biasExists {
		logger.Debug("Loading the biases")
		sBias, biasValues, err := fetchTensor(redisClient, biasName)
		if err != nil {
			return nil, err
		}

		// Cast the shape to an int array and build the layer tensor
		dimBias := shapeToIntArray(sBias...)
		// Build the actual tensor
		b = tensor.New(tensor.WithShape(dimBias...), tensor.WithBacking(biasValues))
		hasBias = true
	}

	return &Layer{
		Name:    name,
		Weights: w,
		HasBias: hasBias,
		Bias:    b,
	}, nil

}

// update the layer given a particular gradient using SGD and the given learning rate
func (layer *Layer) update(g *Gradient, lr float32) error {

	// update the gradients with the learning rate
	err := g.applyLR(lr)
	if err != nil {
		return err
	}

	// Subtract the gradients from the layer
	layer.Weights, _ = layer.Weights.Sub(g.Weights)

	// Just update if the bias is set
	if layer.HasBias {
		layer.Bias, _ = layer.Bias.Sub(g.Bias)
	}

	return nil
}

// Reads a gradient from the database
func newGradient(redisClient *redisai.Client, layerName, psId string, funcId int) (*Gradient, error) {

	fId := strconv.Itoa(funcId)

	// Get the redis keys
	weightName, biasName := getWeightKeys(layerName, true, psId, fId)

	// Build the weight tensor
	sWeights, weightValues, err := fetchTensor(redisClient, weightName)
	if err != nil {
		return nil, err
	}

	dimWeights := shapeToIntArray(sWeights...)
	w := tensor.New(tensor.WithShape(dimWeights...), tensor.WithBacking(weightValues))

	// If we have to build the bias tensor
	var b *tensor.Dense
	biasExists, err := tensorExists(redisClient, biasName)
	if err != nil {
		return nil, err
	}

	hasBias := false
	if biasExists {
		// Build the bias tensor
		sBias, biasValues, err := fetchTensor(redisClient, biasName)
		if err != nil {
			return nil, err
		}

		dimBias := shapeToIntArray(sBias...)
		// Build the actual tensor
		b = tensor.New(tensor.WithShape(dimBias...), tensor.WithBacking(biasValues))
		hasBias = true
	}

	return &Gradient{
		Weights: w,
		HasBias: hasBias,
		Bias:    b,
	}, nil

}

// Multiplies the weights and bias by the learning rate before applying it to a Layer in an update
func (g *Gradient) applyLR(lr float32) error {

	var err error
	g.Weights, err = g.Weights.MulScalar(lr, false)
	g.Bias, err = g.Bias.MulScalar(lr, false)

	if err != nil {
		return err
	}

	return nil
}

// Sets the model learning rate to the new value
func (lrs LrScheduler) updateLr(m *Model) {
	m.logger.Info("Updating the LR",
		zap.Float32("Rate", lrs.rate),
		zap.Float32("Current rate", m.lr))
	m.lr *= lrs.rate
}

// tensorExists simply returns whether the tensor is present in the cache
// In some networks (i.e resnets) the bias of the layers is not used, so in those
// cases it will not be published. In this case we can see whether that is true
func tensorExists(client *redisai.Client, tensorName string) (bool, error) {
	res, err := redis.Int(client.DoOrSend("EXISTS", redis.Args{tensorName}, nil))
	if err != nil {
		return false, err
	}

	// we get a 1 if it exists and a 0 if it doesn't
	switch res {
	case 0:
		return false, err
	case 1:
		return true, nil
	default:
		return false, fmt.Errorf("received unknown result from the cache: %v", res)
	}

}
