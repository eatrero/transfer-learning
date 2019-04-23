import React, { Component } from "react";
import "./App.css";
import { Webcam } from "./webcam";
import Button from "@material-ui/core/Button";
import styled from "styled-components";
import { model, train, predict, stopPredicting } from "./model";
import { ControllerDataset } from "./controller_dataset";
import * as tf from "@tensorflow/tfjs";

// The number of classes we want to predict. In this example, we will be
// predicting 4 classes (one for each of the cards)
const NUM_CLASSES = 4;

const WebcamBoxOuter = styled.div`
  margin: 40px;
`;

const WebcamBoxInner = styled.div`
  border: 1px solid white;
`;

const Class = styled.div`
  height: 150px;
  background-color: grey;
  border: ${prop =>
    prop.isPredicted ? "3px solid white" : "3px solid #00cdbe"};
`;
const Class1 = styled(Class)``;
const Class2 = styled(Class)``;
const Class3 = styled(Class)``;
const Class4 = styled(Class)``;

const Heading = styled.div`
  display: grid;
  grid-gap: 10px;
  grid-template-columns: [col1-start] 400px [col2-start] 224px [col2-end];
`;

const Controls = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: space-between;
`;

const Buttons = styled.div`
  display: flex;
  justify-content: space-around;
  height: 40px;
`;

const TextArea = styled.textarea`
  height: 240px;
`;

const Layout = styled.div`
  display: grid;
  grid-gap: 10px;
  grid-template-columns: [col1-start] 400px [col2-start] 230px [col2-end];
  grid-template-rows: [row1-start] 155px [row2-start] 155px [row3-start] 155px [row4-start] 155px [row4-end];

  ${WebcamBoxOuter} {
    grid-column: col1-start;
    grid-row: row1-start / row3-start;
  }

  ${Controls} {
    grid-column: col1-start;
    grid-row: row3-start / row4-end;
  }

  ${Class1} {
    grid-column: col2-start / col2-end;
    grid-row: row1-start;
  }

  ${Class2} {
    grid-column: col2-start / col2-end;
    grid-row: row2-start;
  }

  ${Class3} {
    grid-column: col2-start / col2-end;
    grid-row: row3-start;
  }

  ${Class4} {
    grid-column: col2-start / col2-end;
    grid-row: row4-start / row4-end;
  }
`;

class App extends Component {
  constructor(props) {
    super(props);
    this.myRef0 = React.createRef();
    this.myRef1 = React.createRef();
    this.myRef2 = React.createRef();
    this.myRef3 = React.createRef();
    this.state = { logs: "Initializing..." };
  }

  async componentDidMount() {
    this.webcam = new Webcam(document.getElementById("webcam"));
    this.controllerDataset = new ControllerDataset(NUM_CLASSES);

    document.getElementById("no-webcam").style.display = "none";
    try {
      await this.webcam.setup();
    } catch (e) {
      document.getElementById("no-webcam").style.display = "block";
    }

    const mobilenet = await tf.loadModel(
      "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json",
    );

    // Return a model that outputs an internal activation.
    const layer = mobilenet.getLayer("conv_pw_13_relu");
    this.mobilenet = tf.model({
      inputs: mobilenet.inputs,
      outputs: layer.output,
    });

    // Warm up the model
    tf.tidy(() => mobilenet.predict(this.webcam.capture()));

    this.featureExtract = tf.model({
      inputs: this.mobilenet.inputs,
      outputs: layer.output,
    });

    // Get classifier
    this.classifier = model(NUM_CLASSES);
    this.currentActiveClass = 1;
  }

  onClickTrain = async () => {
    await tf.nextFrame();
    await tf.nextFrame();
    stopPredicting();
    train(this.classifier, this.controllerDataset, this.trainCallback);
  };

  trainCallback = log => {
    this.setState({
      logs: `${this.state.logs}\nTraining batch. Loss: ${log.loss}.`,
    });
  };

  onClickAddSampleImg = label => {
    tf.tidy(() => {
      const img = this.webcam.capture();
      this.controllerDataset.addExample(this.mobilenet.predict(img), label);

      // Draw the preview thumbnail.
      this.drawThumb(img, label);
    });
    this.setState({
      logs: `${this.state.logs}\nAdded sample for class ${label}.`,
    });
  };

  predictCallback = classId => {
    this.setState({
      prediction: classId,
    });
  };

  onClickPredict = async () => {
    predict(this.webcam, this.mobilenet, this.classifier, this.predictCallback);
  };

  drawThumb = (img, label) => {
    let ref;
    if (label === 0) {
      ref = this.myRef0;
    } else if (label === 1) {
      ref = this.myRef1;
    } else if (label === 2) {
      ref = this.myRef2;
    } else {
      ref = this.myRef3;
    }

    this.draw(img, ref.current.firstElementChild);
  };

  draw = (image, canvas) => {
    const [width, height] = [224, 224];
    const ctx = canvas.getContext("2d");
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
      const j = i * 4;
      imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
      imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
      imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
      imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
  };

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <div id="no-webcam">
            No webcam found. <br />
            To use this demo, use a device with a webcam.
          </div>

          <Heading>
            <h2>Input</h2>
            <h2>Classes</h2>
          </Heading>

          <Layout>
            <WebcamBoxOuter>
              <WebcamBoxInner>
                <video
                  autoPlay
                  playsInline
                  muted
                  id="webcam"
                  width="224"
                  height="224"
                />
              </WebcamBoxInner>
            </WebcamBoxOuter>

            <Controls>
              <Buttons>
                <Button
                  onClick={() => this.onClickTrain()}
                  variant="contained"
                  color="primary"
                >
                  Train
                </Button>

                <Button
                  onClick={() => this.onClickPredict()}
                  variant="contained"
                  color="secondary"
                >
                  Predict
                </Button>
              </Buttons>
              <TextArea rows={6} value={this.state.logs} />
            </Controls>

            <Class1
              ref={this.myRef0}
              onClick={() => this.onClickAddSampleImg(0, this.myRef0)}
              isPredicted={this.state.prediction === 0}
            >
              <canvas />
            </Class1>
            <Class2
              ref={this.myRef1}
              onClick={() => this.onClickAddSampleImg(1, this.myRef1)}
              isPredicted={this.state.prediction === 1}
            >
              <canvas />
            </Class2>
            <Class3
              ref={this.myRef2}
              onClick={() => this.onClickAddSampleImg(2, this.myRef2)}
              isPredicted={this.state.prediction === 2}
            >
              <canvas />
            </Class3>
            <Class4
              ref={this.myRef3}
              onClick={() => this.onClickAddSampleImg(3, this.myRef3)}
              isPredicted={this.state.prediction === 3}
            >
              <canvas />
            </Class4>
          </Layout>
        </header>
      </div>
    );
  }
}

export default App;
