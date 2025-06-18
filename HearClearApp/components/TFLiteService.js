// services/TFLiteService.js
import Tflite from "react-native-tflite";

const tflite = new Tflite();

export const loadModel = () => {
  return new Promise((resolve, reject) => {
    tflite.loadModel(
      {
        model: "model.tflite",
        numThreads: 1,
      },
      (err, res) => {
        if (err) reject(err);
        else resolve(res);
      }
    );
  });
};

export const runInference = (inputArray) => {
  return new Promise((resolve, reject) => {
    tflite.runModelOnArray(
      {
        input: inputArray,
        inputShape: [1, inputArray.length], // Tùy theo model của bạn
        outputShape: [1, N], // N tùy mô hình output
        type: "float32",
      },
      (err, res) => {
        if (err) reject(err);
        else resolve(res);
      }
    );
  });
};
