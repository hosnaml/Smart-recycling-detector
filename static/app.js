document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('camera-stream');
    const canvas = document.getElementById('snapshot-canvas');
    const captureBtn = document.getElementById('capture-btn');
    const scanningOverlay = document.getElementById('scanning-overlay');
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const resultDesc = document.getElementById('result-desc');
    const resultIcon = document.getElementById('result-icon');
    
    // Q&A Elements
    const qaSection = document.getElementById('qa-section');
    const qaQuestion = document.getElementById('qa-question');
    const qaOptionsContainer = document.getElementById('qa-options');
    const finalInstruction = document.getElementById('final-instruction');

    let isScanning = false;
    let inQAState = false;
    let currentOptionsMap = {}; // Maps 'ArrowLeft' -> action/instruction

    // Interactive Q&A Graph/Dictionary (Swedish Context)
    const QA_DATABASE = {
        'Plastic': {
            question: "What type of plastic item is this?",
            options: [
                { direction: 'ArrowLeft', label: 'Bottle (Pant)', text: 'Take this to a supermarket reverse vending machine (Pant) to get your deposit back!' },
                { direction: 'ArrowRight', label: 'Food Tray', text: 'Empty and rinse out any food residue, then place in the plastic recycling bin.' },
                { direction: 'ArrowUp', label: 'Soft Plastic', text: 'Ensure it is empty and place it in the plastic recycling bin.' }
            ]
        },
        'Metal': {
            question: "What kind of metal item is it?",
            options: [
                { direction: 'ArrowLeft', label: 'Can (Pant)', text: 'Do not crush it! Take it to a supermarket Pant station.' },
                { direction: 'ArrowRight', label: 'Food Can', text: 'Rinse it out, fold the sharp lid inside, and put in the metal bin.' },
                { direction: 'ArrowUp', label: 'Alu Foil', text: 'Rinse off any food, crumple it into a ball, and recycle as metal.' }
            ]
        },
        'Glass': {
            question: "What type of glass is it?",
            options: [
                { direction: 'ArrowLeft', label: 'Clear', text: 'Remove the cap/lid and place in the Clear Glass (Ofärgat glas) bin.' },
                { direction: 'ArrowRight', label: 'Colored', text: 'Remove the cap/lid and place in the Colored Glass (Färgat glas) bin.' },
                { direction: 'ArrowUp', label: 'Drinking/Mirror', text: 'Wait! Drinking glasses and mirrors cannot be recycled as packaging. Put in general waste.' }
            ]
        },
        'Paper': {
            question: "Is the paper clean and dry?",
            options: [
                { direction: 'ArrowLeft', label: 'Yes', text: 'Great! Place it in the standard paper recycling bin.' },
                { direction: 'ArrowRight', label: 'No (Greasy)', text: 'Place soiled paper (like napkins) in the compost/organic bin or general waste.' }
            ]
        },
        'Cardboard': {
            question: "Is this a greasy pizza box or clean cardboard?",
            options: [
                { direction: 'ArrowLeft', label: 'Pizza Box', text: 'Tear off the clean lid for cardboard recycling. The greasy bottom goes in general waste.' },
                { direction: 'ArrowRight', label: 'Clean Box', text: 'Flatten the box completely to save space, then place in cardboard recycling.' }
            ]
        }
    };

    // Helper to get symbol for direction
    const getDirectionSymbol = (dir) => {
        switch(dir) {
            case 'ArrowLeft': return '⬅️ Left';
            case 'ArrowRight': return '➡️ Right';
            case 'ArrowUp': return '⬆️ Up';
            case 'ArrowDown': return '⬇️ Down';
            default: return dir;
        }
    };

    // Icons map for different categories
    const categoryIcons = {
        'Plastic': '🧴',
        'Paper': '📄',
        'Glass': '🍾',
        'Metal': '🥫',
        'Cardboard': '📦',
        'Organic': '🍎'
    };

    // Check Backend ML Status and Show Toast Notification
    fetch('/status')
        .then(res => res.json())
        .then(data => {
            const toastEl = document.getElementById('statusToast');
            const toastMsg = document.getElementById('toastMessage');
            
            toastMsg.textContent = data.message;
            if (data.ml_ready) {
                toastEl.classList.add('bg-success');
            } else {
                toastEl.classList.add('bg-warning', 'text-dark');
                toastEl.classList.remove('text-white');
                toastEl.querySelector('.btn-close').classList.remove('btn-close-white');
            }
            
            const toast = new bootstrap.Toast(toastEl, { delay: 6000 });
            toast.show();
        })
        .catch(err => console.error("Error fetching model status:", err));

    // iPhone Continuity Camera integration
    let currentStream = null;

    async function initializeCamera() {
        try {
            // Request permission so camera names are available.
            const permissionStream =
                await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });

            // Stop temporary camera before selecting the preferred one.
            permissionStream.getTracks().forEach(track => track.stop());

            const devices =
                await navigator.mediaDevices.enumerateDevices();

            const cameras = devices.filter(
                device => device.kind === 'videoinput'
            );

            console.log(
                'Available cameras:',
                cameras.map(camera => camera.label)
            );

            // Select the normal iPhone camera, not Desk View.
            const iphoneCamera = cameras.find(camera => {
                const name = camera.label.toLowerCase();

                return name.includes('iphone') &&
                       name.includes('camera') &&
                       !name.includes('desk view');
            });

            const constraints = iphoneCamera
                ? {
                    video: {
                        deviceId: {
                            exact: iphoneCamera.deviceId
                        }
                    },
                    audio: false
                }
                : {
                    video: true,
                    audio: false
                };

            currentStream =
                await navigator.mediaDevices.getUserMedia(constraints);

            video.srcObject = currentStream;

            await video.play();

            if (iphoneCamera) {
                console.log('Using iPhone Camera');
            } else {
                console.log('Using default webcam');
            }

        } catch (error) {
            console.error('Camera initialization error:', error);

            alert(
                'Unable to access the camera. Check camera permissions and Continuity Camera.'
            );
        }
    }

    initializeCamera();

    // 2. Capture and Send to Backend
    const captureAndDetect = async () => {
        if (isScanning || !video.srcObject) return;
        isScanning = true;

        // UI: Show loading state, hide previous result
        scanningOverlay.style.setProperty('display', 'flex', 'important');
        resultCard.style.display = 'none';

        // Draw current video frame to hidden canvas
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;

        const cropSize = Math.floor(
            Math.min(videoWidth, videoHeight) * 0.55
        );

        const cropX = Math.floor((videoWidth - cropSize) / 2);
        const cropY = Math.floor((videoHeight - cropSize) / 2);

        canvas.width = 224;
        canvas.height = 224;

        const ctx = canvas.getContext('2d');

        ctx.drawImage(
            video,
            cropX, cropY, cropSize, cropSize,
            0, 0, 224, 224
        );
        
        // Convert to base64 JPEG
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) throw new Error("Server error");
            const result = await response.json();
            
            // UI: Display Result
            resultIcon.textContent = categoryIcons[result.label] || '📦';
            resultTitle.textContent = result.label;
            resultTitle.className = `fw-bold mb-2 display-6 text-${result.color}`;
            
            // Reset visibility
            resultDesc.style.display = 'none';
            qaSection.style.display = 'none';
            finalInstruction.style.display = 'none';
            qaOptionsContainer.innerHTML = '';
            currentOptionsMap = {};

            // Check if we have interactive Q&A for this material
            if (QA_DATABASE[result.label]) {
                inQAState = true;
                const qaData = QA_DATABASE[result.label];
                qaQuestion.textContent = qaData.question;
                
                qaData.options.forEach(opt => {
                    currentOptionsMap[opt.direction] = opt.text;
                    
                    const col = document.createElement('div');
                    col.className = 'col-sm-6 col-md-4';
                    col.innerHTML = `
                        <div class="qa-option-card text-center" id="opt-${opt.direction}">
                            <span class="joystick-hint">${getDirectionSymbol(opt.direction)}</span>
                            <h6 class="mb-0 fw-bold mt-2">${opt.label}</h6>
                        </div>
                    `;
                    qaOptionsContainer.appendChild(col);
                });
                
                qaSection.style.display = 'block';
            } else {
                // Standard fallback (e.g. Organic, Trash)
                inQAState = false;
                resultDesc.textContent = result.instructions;
                resultDesc.style.display = 'block';
            }
            
            // Trigger reflow to restart CSS animation
            resultCard.classList.remove('result-pop');
            void resultCard.offsetWidth; 
            resultCard.classList.add('result-pop');
            resultCard.style.display = 'block';

        } catch (error) {
            console.error("Error during detection:", error);
            alert("Detection failed. Please check the backend connection.");
        } finally {
            // UI: Hide loading state
            scanningOverlay.style.setProperty('display', 'none', 'important');
            isScanning = false;
        }
    };

    // 3. Setup Triggers

    // Manual Button Click
    captureBtn.addEventListener('click', captureAndDetect);

    // Keyboard Fallback (Enter key to scan, Arrows to answer Q&A)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            // If in Q&A state, reset and scan again. Otherwise, just scan.
            if (inQAState) {
                inQAState = false;
                resultCard.style.display = 'none';
            }
            captureAndDetect();
        } else if (inQAState && currentOptionsMap[e.key]) {
            e.preventDefault();
            
            // Highlight selected option visually
            document.querySelectorAll('.qa-option-card').forEach(el => el.classList.remove('active'));
            const selectedCard = document.getElementById(`opt-${e.key}`);
            if (selectedCard) selectedCard.classList.add('active');
            
            // Display final instruction
            finalInstruction.textContent = currentOptionsMap[e.key];
            finalInstruction.style.display = 'block';
            
            // End Q&A state so arrows don't trigger again until next scan
            inQAState = false;
        }
    });

    // Server-Sent Events (Hardware Joystick trigger via backend)
    const evtSource = new EventSource("/events");
    evtSource.onmessage = function(event) {
        const joystickKeys = {
            "LEFT": "ArrowLeft",
            "RIGHT": "ArrowRight",
            "UP": "ArrowUp",
            "CAPTURE": "Enter"
        };

        const key = joystickKeys[event.data];

        if (key) {
            console.log("Hardware joystick event:", event.data);
            document.dispatchEvent(new KeyboardEvent("keydown", {
                key: key,
                bubbles: true
            }));
        }
    };
    
    evtSource.onerror = function(err) {
        console.error("EventSource failed:", err);
    };
});
