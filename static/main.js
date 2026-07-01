document.addEventListener('DOMContentLoaded', () => {
    // Fetch and display active backend device status
    fetch('/api/device')
        .then(res => res.json())
        .then(data => {
            document.getElementById('device-badge').innerText = `DEVICE: ${data.device}`;
        })
        .catch(err => console.error('Error fetching device status:', err));

    // ------------------ TAB SYSTEM ------------------
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const target = tab.getAttribute('data-tab');
            document.getElementById(target).classList.add('active');
        });
    });

    // Helper: update element class list
    function setClass(element, removeClass, addClass) {
        element.classList.remove(removeClass);
        element.classList.add(addClass);
    }

    // ------------------ STEGANALYSIS TAB ------------------
    const scanDropzone = document.getElementById('scan-dropzone');
    const scanInput = document.getElementById('scan-file-input');
    const scanPreview = document.getElementById('scan-preview');
    const btnScan = document.getElementById('btn-scan');
    
    const scanResultBox = document.getElementById('scan-result-box');
    const scanResultStatus = document.getElementById('scan-result-status');
    const scanResultConf = document.getElementById('scan-result-conf');
    const gradcamImg = document.getElementById('gradcam-img');
    const featuresImg = document.getElementById('features-img');
    const featuresContainer = document.getElementById('features-container');

    // Trigger file dialog
    scanDropzone.addEventListener('click', () => scanInput.click());

    // File selection preview
    scanInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                scanPreview.src = event.target.result;
                scanPreview.style.display = 'block';
                document.getElementById('scan-dropzone-inner').style.display = 'none';
                btnScan.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });

    // Run Scan
    btnScan.addEventListener('click', async () => {
        const file = scanInput.files[0];
        if (!file) return;

        btnScan.disabled = true;
        btnScan.innerHTML = '⚡ Scanning image...';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Update Prediction card
                const isStego = data.prediction.includes('Stego');
                scanResultBox.className = 'result-box ' + (isStego ? 'stego' : 'clean');
                scanResultStatus.innerText = (isStego ? '🚨 ' : '🛡️ ') + data.prediction.toUpperCase();
                scanResultConf.innerText = `Inference Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                scanResultBox.style.display = 'block';

                // Render Grad-CAM
                gradcamImg.src = data.gradcam_url;
                document.getElementById('gradcam-placeholder-text').style.display = 'none';
                gradcamImg.style.display = 'block';
                
                // Render Features
                if (data.features_url) {
                    featuresImg.src = data.features_url;
                    document.getElementById('features-placeholder-text').style.display = 'none';
                    featuresImg.style.display = 'block';
                    featuresContainer.style.display = 'block';
                }
            } else {
                alert(`Analysis failed: ${data.error}`);
            }
        } catch (error) {
            console.error('Scan Error:', error);
            alert('Failed to connect to backend steganalysis server.');
        } finally {
            btnScan.disabled = false;
            btnScan.innerHTML = '🔎 Run UC-DFNet Steganalysis';
        }
    });


    // ------------------ SANDBOX TAB: EMBEDDING ------------------
    const covDropzone = document.getElementById('cov-dropzone');
    const covInput = document.getElementById('cov-file-input');
    const covPreview = document.getElementById('cov-preview');
    
    const embedAlgo = document.getElementById('embed-algo');
    const embedChannels = document.getElementById('embed-channels');
    const embedParam = document.getElementById('embed-param');
    const embedMessage = document.getElementById('embed-message');
    const btnEmbed = document.getElementById('btn-embed');
    const embedResultBox = document.getElementById('embed-result-box');
    const btnDownloadStego = document.getElementById('btn-download-stego');

    let currentStegoUrl = null;

    covDropzone.addEventListener('click', () => covInput.click());

    covInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                covPreview.src = event.target.result;
                covPreview.style.display = 'block';
                document.getElementById('cov-dropzone-inner').style.display = 'none';
                btnEmbed.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });

    btnEmbed.addEventListener('click', async () => {
        const file = covInput.files[0];
        const message = embedMessage.value;
        if (!file || !message) return;

        btnEmbed.disabled = true;
        btnEmbed.innerHTML = '🔒 Embedding payload...';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('message', message);
        formData.append('algo', embedAlgo.value);
        formData.append('channels', embedChannels.value);
        formData.append('param', embedParam.value);

        try {
            const response = await fetch('/api/embed', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.success) {
                currentStegoUrl = data.download_url;
                embedResultBox.style.display = 'block';
            } else {
                alert(`Embedding failed: ${data.error}`);
            }
        } catch (error) {
            console.error('Embedding Error:', error);
            alert('Failed to connect to backend steganography encoder.');
        } finally {
            btnEmbed.disabled = false;
            btnEmbed.innerHTML = '🔒 Embed Payload';
        }
    });

    btnDownloadStego.addEventListener('click', () => {
        if (currentStegoUrl) {
            // Trigger actual download of stego image
            const link = document.createElement('a');
            link.href = currentStegoUrl;
            link.download = 'stego_image.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });


    // ------------------ SANDBOX TAB: EXTRACTION ------------------
    const stegDropzone = document.getElementById('steg-dropzone');
    const stegInput = document.getElementById('steg-file-input');
    const stegPreview = document.getElementById('steg-preview');
    
    const extractAlgo = document.getElementById('extract-algo');
    const extractChannels = document.getElementById('extract-channels');
    const extractParam = document.getElementById('extract-param');
    const btnExtract = document.getElementById('btn-extract');
    const extractedMessageConsole = document.getElementById('extracted-message-console');

    stegDropzone.addEventListener('click', () => stegInput.click());

    stegInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                stegPreview.src = event.target.result;
                stegPreview.style.display = 'block';
                document.getElementById('steg-dropzone-inner').style.display = 'none';
                btnExtract.disabled = false;
            };
            reader.readAsDataURL(file);
        }
    });

    btnExtract.addEventListener('click', async () => {
        const file = stegInput.files[0];
        if (!file) return;

        btnExtract.disabled = true;
        btnExtract.innerHTML = '🔓 Decoding...';
        extractedMessageConsole.innerText = 'Extracting, please wait...';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('algo', extractAlgo.value);
        formData.append('channels', extractChannels.value);
        formData.append('param', extractParam.value);

        try {
            const response = await fetch('/api/extract', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.success) {
                if (data.message) {
                    extractedMessageConsole.innerText = data.message;
                } else {
                    extractedMessageConsole.innerText = '(No message detected or null string decoded)';
                }
            } else {
                extractedMessageConsole.innerText = `Extraction failed: ${data.error}`;
            }
        } catch (error) {
            console.error('Extraction Error:', error);
            extractedMessageConsole.innerText = 'Failed to connect to backend extractor.';
        } finally {
            btnExtract.disabled = false;
            btnExtract.innerHTML = '🔓 Extract Message';
        }
    });


    // ------------------ TRAINING TAB ------------------
    const trainSamples = document.getElementById('train-samples');
    const trainEpochs = document.getElementById('train-epochs');
    const trainBatch = document.getElementById('train-batch');
    const btnTrain = document.getElementById('btn-train');
    
    // Kaggle Loader Controls
    const btnKaggle = document.getElementById('btn-kaggle');
    const kaggleLimit = document.getElementById('kaggle-limit');
    
    const trainConsole = document.getElementById('train-console');
    const progressBar = document.getElementById('progress-bar');
    
    let isTraining = false;
    let activeAction = null; // 'train' or 'kaggle'
    let pollInterval = null;

    btnTrain.addEventListener('click', async () => {
        if (isTraining) return;

        const samples = parseInt(trainSamples.value);
        const epochs = parseInt(trainEpochs.value);
        const batch = parseInt(trainBatch.value);

        if (isNaN(samples) || isNaN(epochs) || isNaN(batch)) {
            alert('Please enter valid numerical training configuration.');
            return;
        }

        btnTrain.disabled = true;
        btnKaggle.disabled = true;
        btnTrain.innerHTML = '⚡ Training...';
        trainConsole.innerText = 'Initializing training pipeline...';
        progressBar.style.width = '5%';
        isTraining = true;
        activeAction = 'train';

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ samples, epochs, batch })
            });
            const data = await response.json();

            if (data.success) {
                pollInterval = setInterval(pollStatus, 1500);
            } else {
                alert(`Failed to start training: ${data.error}`);
                btnTrain.disabled = false;
                btnKaggle.disabled = false;
                btnTrain.innerHTML = '🚀 Start Local Training';
                isTraining = false;
                activeAction = null;
            }
        } catch (error) {
            console.error('Train trigger failed:', error);
            alert('Failed to connect to backend training process.');
            btnTrain.disabled = false;
            btnKaggle.disabled = false;
            btnTrain.innerHTML = '🚀 Start Local Training';
            isTraining = false;
            activeAction = null;
        }
    });

    btnKaggle.addEventListener('click', async () => {
        if (isTraining) return;

        const limit = parseInt(kaggleLimit.value);
        if (isNaN(limit) || limit < 10) {
            alert('Please enter a valid limit number (minimum 10).');
            return;
        }

        btnKaggle.disabled = true;
        btnTrain.disabled = true;
        btnKaggle.innerHTML = '⚡ Fetching Kaggle...';
        trainConsole.innerText = 'Connecting to Kaggle API to load dataset...';
        progressBar.style.width = '5%';
        isTraining = true;
        activeAction = 'kaggle';

        try {
            const response = await fetch('/api/kaggle_load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ limit })
            });
            const data = await response.json();

            if (data.success) {
                pollInterval = setInterval(pollStatus, 1500);
            } else {
                alert(`Kaggle load failed: ${data.error}`);
                btnKaggle.disabled = false;
                btnTrain.disabled = false;
                btnKaggle.innerHTML = '📥 Fetch & Integrate Kaggle Dataset';
                isTraining = false;
                activeAction = null;
            }
        } catch (error) {
            console.error('Kaggle trigger error:', error);
            alert('Failed to trigger Kaggle downloader.');
            btnKaggle.disabled = false;
            btnTrain.disabled = false;
            btnKaggle.innerHTML = '📥 Fetch & Integrate Kaggle Dataset';
            isTraining = false;
            activeAction = null;
        }
    });

    async function pollStatus() {
        try {
            const response = await fetch('/api/train_status');
            const data = await response.json();

            if (data.logs) {
                trainConsole.innerText = data.logs;
                trainConsole.scrollTop = trainConsole.scrollHeight;
            }

            if (data.progress) {
                progressBar.style.width = `${data.progress}%`;
            }

            if (!data.running) {
                clearInterval(pollInterval);
                isTraining = false;
                progressBar.style.width = '100%';
                
                if (activeAction === 'train') {
                    btnTrain.disabled = false;
                    btnKaggle.disabled = false;
                    btnTrain.innerHTML = '🚀 Start Local Training';
                    alert('Training Wizard complete! The local model weights have been updated.');
                } else if (activeAction === 'kaggle') {
                    btnKaggle.disabled = false;
                    btnTrain.disabled = false;
                    btnKaggle.innerHTML = '📥 Fetch & Integrate Kaggle Dataset';
                    alert('Kaggle Dataset download and integration complete! You can now proceed to training.');
                }
                activeAction = null;
            }
        } catch (error) {
            console.error('Polling Error:', error);
        }
    }
});
