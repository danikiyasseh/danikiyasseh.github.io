const path_to_model = './model.json'

// import {IMAGENET_CLASSES} from './imagenet_classes';

// let net;
// const TOPK_PREDICTIONS = 5

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
  
  //const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  //showResults(imgEl, classes);
  console.log(logits);
}

app();
