import threading
import time

class HardwareInput:
    def __init__(self, trigger_callback):
        self.trigger_callback = trigger_callback
        self.running = False
        self.thread = None
        try:
            from sense_hat import SenseHat
            self.sense = SenseHat()
            self.has_sense = True
            print("Sense HAT found and initialized.")
        except ImportError:
            self.sense = None
            self.has_sense = False
            print("Sense HAT not found. Running in mock/fallback mode. Use keyboard 'Enter' on the web interface to trigger.")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            if self.has_sense:
                # Polling the joystick events
                for event in self.sense.stick.get_events():
                    if event.action == "pressed" and event.direction == "middle":
                        print("Hardware Joystick pressed!")
                        self.trigger_callback()
            time.sleep(0.1)

    def stop(self):
        self.running = False
