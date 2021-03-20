const path_to_model = './MobileNetV2/model.json'

// import {IMAGENET_CLASSES} from './imagenet_classes';

// let net;
// const TOPK_PREDICTIONS = 5

const topk = 3;
const IMAGE_HEIGHT = 224;
const IMAGE_WIDTH = 224;
let model;

const app = async () => {
  status('Loading model...');
  //console.log('Loading model..');
  
  // Load the model.
  // net = await mobilenet.load();
  model = await tf.loadLayersModel(path_to_model);
  status('Successfully loaded model!');
  // console.log('Successfully loaded model');
  
  // Forward pass with zeros to warm-up model (faster processing later).
  model.predict(tf.zeros([1,IMAGE_HEIGHT,IMAGE_WIDTH,3])).dispose();
  
  // Get image.
  const imgEl = document.getElementById('img');
  // If image available, predict. Otherwise, wait for loaded image.
  if (imgEl.complete && imgEl.naturalHeight !== 0) {
    console.log('Found image...');
    predict(imgEl, topk);
    imgEl.style.display = '';
  } else {
    console.log('waiting for image...');
    imgEl.onload = () => {
      predict(imgEl, topk);
      imgEl.style.display = '';
    }
  }  
  // Show image that was just loaded
  document.getElementById('file-container').style.display = '';
  
//   // Make a prediction through the model on our image.
//   // const result = await net.classify(imgEl);
//   const logits = await model.predict(imgSample);
//   console.log(logits);
}

async function predict(imgEl, topk) {  
  const preds = await classify(imgEl, topk)
  
  status('Done!');
  console.log(preds);
  
  showResults(imgEl, preds)
  console.log('Showed Results') 
}

// Function to make prediction and obtain topk results.
async function classify(imgEl, topk) {
    status('Predicting...');
  
    const logits = tf.tidy(() => {
          // Load image into TFJS world.
          const img = tf.browser.fromPixels(imgEl).toFloat();
          // Normalize the image from [0-255] to [0-1].
          const offset = 255;
          const normalized = img.div(offset);
          // Reshape image for network.
          const imgSample = normalized.reshape([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]);

          return model.predict(imgSample);
          //logits.dispose();
          //return classes;
    });
  
    const classes = await getTopKClasses(logits, topk);
    return classes;
}

async function getTopKClasses(logits, topK) {
  const softmax = tf.softmax(logits);
  const values = await softmax.data();
  softmax.dispose();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: UNESCO_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    });
  }
  return topClassesAndProbs;
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

  predictionsElement.insertBefore(
      predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    console.log('Hello')
    // Only process image files (skip non image files)
    // if (!f.type.match('image.*')) {
    //   console.log('skipped');
    //   continue;
    // }
    let reader = new FileReader();
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_WIDTH;
      img.height = IMAGE_HEIGHT;
      //console.log(img.height);
      img.onload = () => predict(img, topk);
      //console.log(img.onload);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const UNESCO_CLASSES = {
 0: 'Abu-Mena-UNESCO-site',
 1: 'Aflaj-Irrigation-Systems-of-Oman-UNESCO-site',
 2: 'Ahwar-of-Southern-Iraq-UNESCO-site',
 3: 'Al-Ahsa-Oasis-UNESCO-site',
 4: 'Al-Ain-UNESCO-site',
 5: 'Al-Balad,-Jeddah-UNESCO-site',
 6: 'Al-Maghtas-UNESCO-site',
 7: 'Al-Zubarah-UNESCO-site',
 8: 'Amphitheatre-of-El-Jem-UNESCO-site',
 9: 'Ancient-City-of-Bosra-UNESCO-site',
 10: 'Ancient-City-of-Damascus-UNESCO-site',
 11: 'Ancient-Ksour-of-Ouadane,-Chinguetti,-Tichitt-and-Oualata-UNESCO-site',
 12: 'Anjar,-Lebanon-UNESCO-site',
 13: 'Archaeological-Site-of-Carthage-UNESCO-site',
 14: 'Archaeological-Sites-of-Bat,-Al-Khutm-and-Al-Ayn-UNESCO-site',
 15: 'Assur-UNESCO-site',
 16: 'Baalbek-UNESCO-site',
 17: 'Babylon-UNESCO-site',
 18: 'Bahla-Fort-UNESCO-site',
 19: 'Bahrain-Pearling-Trail-UNESCO-site',
 20: 'Battir-UNESCO-site',
 21: 'Beni-Hammad-Fort-UNESCO-site',
 22: 'Byblos-UNESCO-site',
 23: 'Casbah-of-Algiers-UNESCO-site',
 24: 'Cedars-of-God-UNESCO-site',
 25: 'Church-of-the-Nativity-UNESCO-site',
 26: 'Citadel-of-Arbil-UNESCO-site',
 27: 'Citadel-of-Salah-Ed-Din-UNESCO-site',
 28: 'Cyrene,-Libya-UNESCO-site',
 29: 'Dead-Cities-UNESCO-site',
 30: 'Dilmun-Burial-Mounds-UNESCO-site',
 31: 'Diriyah-UNESCO-site',
 32: 'Djémila-UNESCO-site',
 33: 'Dougga-UNESCO-site',
 34: 'El-Jadida-UNESCO-site',
 35: 'Essaouira-UNESCO-site',
 36: 'Fes-el-Bali-UNESCO-site',
 37: 'Frankincense-Trail-UNESCO-site',
 38: 'Gebel-Barkal-and-the-Sites-of-the-Napatan-Region-UNESCO-site',
 39: 'Ghadames-UNESCO-site',
 40: 'Giza-pyramid-complex-UNESCO-site',
 41: 'Hatra-UNESCO-site',
 42: 'Hebron-UNESCO-site',
 43: 'Ichkeul-National-Park-UNESCO-site',
 44: 'Islamic-Cairo-UNESCO-site',
 45: 'Kadisha-Valley-UNESCO-site',
 46: 'Kairouan-UNESCO-site',
 47: 'Kerkouane-UNESCO-site',
 48: 'Krak-des-Chevaliers-UNESCO-site',
 49: 'Ksar-of-Ait-Ben-Haddou-UNESCO-site',
 50: 'Leptis-Magna-UNESCO-site',
 51: 'Medina-of-Marrakesh-UNESCO-site',
 52: 'Medina-of-Sousse-UNESCO-site',
 53: 'Medina-of-Tunis-UNESCO-site',
 54: 'Meknes-UNESCO-site',
 55: 'Meroë-UNESCO-site',
 56: 'Necropolis-of-Kerkouane-UNESCO-site',
 57: 'Nubian-Monuments-from-Abu-Simbel-to-Philae-UNESCO-site',
 58: 'Old-City-of-Aleppo-UNESCO-site',
 59: 'Petra-UNESCO-site',
 60: 'Qalhat-UNESCO-site',
 61: 'Qasr-Amra-UNESCO-site',
 62: 'Rabat-UNESCO-site',
 63: 'Rock-art-sites-of-Tadrart-Acacus-UNESCO-site',
 64: 'Sabratha-UNESCO-site',
 65: 'Samarra-UNESCO-site',
 66: 'Shibam-UNESCO-site',
 67: 'Site-of-Palmyra-UNESCO-site',
 68: 'Theban-Necropolis-UNESCO-site',
 69: 'Thebes,-Egypt-UNESCO-site',
 70: 'Timgad-UNESCO-site',
 71: 'Tipaza-UNESCO-site',
 72: 'Tyre,-Lebanon-UNESCO-site',
 73: 'Tétouan-UNESCO-site',
 74: 'Umm-ar-Rasas-UNESCO-site',
 75: 'Volubilis-UNESCO-site',
 76: 'Wadi-Al-Hitan-UNESCO-site',
 77: 'Wadi-Rum-UNESCO-site',
 78: 'Zabīd-UNESCO-site'
}

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');
app();
