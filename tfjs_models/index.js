const path_to_model = './model.json'
// const path_to_model = 'https://drive.google.com/file/d/1yeg8usTgWEW1eR9L8YT2o-ZEVPwZ5lje/view?usp=sharing'

// import * as tf from '@tensorflow/tfjs';

import {IMAGENET_CLASSES} from './imagenet_classes';

// let net;
const TOPK_PREDICTIONS = 5

async function app() {
  console.log('Loading model..');

  // Load the model.
  // net = await mobilenet.load();
  const model = await tf.loadLayersModel(path_to_model);
  console.log('Successfully loaded model');

  const imgEl = document.getElementById('img');
  
  const logits = tf.tidy(() => {
    // Load image.
    const img = tf.browser.fromPixels(imgEl);
    const imgSample = img.reshape([1, 224, 224, 3]);

    // Make a prediction through the model on our image.
    // const result = await net.classify(imgEl);
    const logits = await model.predict(imgSample);
    return logits
    // console.log(result);
  });
  
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  showResults(imgEl, classes);
}

async function getTopKClasses(logits, topK) {
  const values = logits.data();
  
  const valuesAndIndices = []
  for (let i = 0; i < values.length(); i++) {
    valuesAndIndices.push({value: values[i],index: i})
  }
  // Sort logits in descending order.
  valuesAndIndices.sort((a,b) => {return b.value - a.value;});
  // Obtain TopK logits and indices.
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);  
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }
  // Obtain TopK logits and classnames.
  const topkClassesAndProbs = [];
  for (let i = 0; i < topK; i++) {
    topkClassesAndProbs.push({className: IMAGENET_CLASSES[topkIndices[i]], probability: topkValues[i]});
  }
  return topkClassesAndProbs
}

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  // predictionsElement.insertBefore(predictionContainer, predictionsElement.firstChild);
}

app();
