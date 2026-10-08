document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('camera-stream');
    const canvas = document.getElementById('snapshot-canvas');
    const captureBtn = document.getElementById('capture-btn');
    const scanningOverlay = document.getElementById('scanning-overlay');
    const resultCard = document.getElementById('result-card');
    const resultTitle = document.getElementById('result-title');
    const resultDesc = document.getElementById('result-desc');
    const resultIcon = document.getElementById('result-icon');

    let isScanning = false;

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

    // 1. Initialize Camera
    // Request rear-facing camera if available
    navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Camera access denied or unavailable", err);
            alert("Unable to access camera. Please allow camera permissions in your browser.");
        });

    // 2. Capture and Send to Backend
    const captureAndDetect = async () => {
        if (isScanning || !video.srcObject) return;
        isScanning = true;

        // UI: Show loading state, hide previous result
        scanningOverlay.style.setProperty('display', 'flex', 'important');
        resultCard.style.display = 'none';

        // Draw current video frame to hidden canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
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
            resultTitle.className = `fw-bold mb-3 display-6 text-${result.color}`;
            resultDesc.textContent = result.instructions;
            
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

    // Keyboard Fallback (Enter key)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            captureAndDetect();
        }
    });

    // Server-Sent Events (Hardware Joystick trigger via backend)
    const evtSource = new EventSource("/events");
    evtSource.onmessage = function(event) {
        if (event.data === "CAPTURE") {
            console.log("Hardware joystick trigger received from backend!");
            captureAndDetect();
        }
    };
    
    evtSource.onerror = function(err) {
        console.error("EventSource failed:", err);
    };
});
