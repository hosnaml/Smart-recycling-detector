from flask import Flask, render_template, request, jsonify, Response
import queue

from input_handler import HardwareInput
from detector import detect_recycling, model

app = Flask(__name__)

event_queue = queue.Queue()


def trigger_capture(direction="middle"):
    """Send joystick events to the browser."""
    events = {
        "middle": "CAPTURE",
        "left": "LEFT",
        "right": "RIGHT", "up": "UP"
    }
    event_queue.put(events[direction])


hw_input = HardwareInput(trigger_callback=trigger_capture)
hw_input.start()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    if model is not None:
        return jsonify({
            "ml_ready": True,
            "message": "ML Model is active and ready for usage!"
        })

    return jsonify({
        "ml_ready": False,
        "message": "Real ML Model not found. Running in Mock fallback mode."
    })


@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON request"}), 400

    image_data = data.get('image')

    if not image_data:
        return jsonify({"error": "No image data provided"}), 400

    result = detect_recycling(image_data)

    if result and result.get("label"):
        hw_input.display_material(result["label"])

    return jsonify(result)


@app.route('/events')
def events():
    def generate():
        while True:
            event = event_queue.get()
            yield f"data: {event}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,
        threaded=True
    )
