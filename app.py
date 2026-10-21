from flask import Flask, render_template, request, jsonify, Response
import queue
from input_handler import HardwareInput
from detector import detect_recycling, model

app = Flask(__name__)

# Queue for Server-Sent Events (SSE) to push hardware triggers to the frontend
event_queue = queue.Queue()

def trigger_capture():
    """Callback function triggered by the hardware input."""
    event_queue.put("CAPTURE")

# Initialize and start the hardware listener in the background
hw_input = HardwareInput(trigger_callback=trigger_capture)
hw_input.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    """Returns the loading status of the ML model."""
    if model is not None:
        return jsonify({"ml_ready": True, "message": "ML Model is active and ready for usage!"})
    else:
        return jsonify({"ml_ready": False, "message": "Real ML Model not found. Running in Mock fallback mode."})

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    image_data = data.get('image')
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400
    
    # Process the image data to determine recycling category
    result = detect_recycling(image_data)
    return jsonify(result)

@app.route('/events')
def events():
    """Endpoint for Server-Sent Events."""
    def generate():
        while True:
            # Block until an event is available in the queue
            event = event_queue.get()
            yield f"data: {event}\n\n"
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True, threaded=True)
