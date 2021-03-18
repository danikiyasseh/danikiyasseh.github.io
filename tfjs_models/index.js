const path_to_model = './model.json'
// const path_to_model = 'https://drive.google.com/file/d/1yeg8usTgWEW1eR9L8YT2o-ZEVPwZ5lje/view?usp=sharing'

// import * as tf from '@tensorflow/tfjs';

// let net;

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  // net = await mobilenet.load();
  const model = await tf.loadLayersModel(path_to_model);
  console.log('Successfully loaded model');

  // Make a prediction through the model on our image.
  const imgEl = document.getElementById('img');
  // const result = await net.classify(imgEl);
  const result = await model.predict(imgEl)
  console.log(result);
}

app();
