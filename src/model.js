import * as tf from "@tensorflow/tfjs";

// Hyperparams
const batchSizeRatio = 0.4;
const epochs = 20;
const hiddenUnits = 100;
const learningRate = 0.0001;

export function model(numClasses) {
  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  let model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({ inputShape: [7, 7, 256] }),
      // Layer 1
      tf.layers.dense({
        units: hiddenUnits,
        activation: "relu",
        kernelInitializer: "varianceScaling",
        useBias: true,
      }),
      // Layer 2. The number of units of the last layer should correspond
      // to the number of classes we want to predict.
      tf.layers.dense({
        units: numClasses,
        kernelInitializer: "varianceScaling",
        useBias: false,
        activation: "softmax",
      }),
    ],
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(learningRate);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });

  return model;
}

/**
 * Sets up and trains the classifier.
 */
export async function train(model, controllerDataset, trainCallback) {
  if (controllerDataset.xs == null) {
    throw new Error("Add some examples before training!");
  }

  const batchSize = Math.floor(controllerDataset.xs.shape[0] * batchSizeRatio);

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        trainCallback(logs);
        await tf.nextFrame();
      },
    },
  });
}

let isPredicting = false;

export function stopPredicting() {
  isPredicting = false;
}

export async function predict(webcam, mobilenet, model, callback) {
  //ui.isPredicting();
  isPredicting = true;
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      // Capture the frame from the webcam.
      const img = webcam.capture();

      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model.
      const activation = mobilenet.predict(img);

      // Make a prediction through our newly-trained model using the activation
      // from mobilenet as input.
      const predictions = model.predict(activation);

      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();

    callback(classId);
    await tf.nextFrame();
  }
}

export default train;
