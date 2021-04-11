  var wavesurfer = WaveSurfer.create({
    container: '#waveform'
    // your options here
    plugins: [
        WaveSurfer.spectrogram.create({
            wavesurfer: wavesurfer,
            container: "#wave-spectrogram",
            labels: true
          })
      ]
  });
  
  wavesurfer.load('./song.mp3');
