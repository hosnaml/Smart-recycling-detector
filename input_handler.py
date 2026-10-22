
import threading
import time


class HardwareInput:

    MATERIAL_COLORS = {
        "cardboard": (161, 98, 7),   # Brown
        "glass": (0, 200, 0),        # Green
        "metal": (255, 200, 0),      # Yellow
        "organic": (120, 255, 0),    # Light green
        "paper": (0, 80, 255),       # Blue
        "plastic": (170, 0, 255)     # Purple
    }

    def __init__(self, trigger_callback):
        self.trigger_callback = trigger_callback
        self.running = False
        self.thread = None

        # Controls automatic LED shutoff
        self.led_timer = None
        self.led_lock = threading.Lock()
        self.led_generation = 0

        try:
            from sense_hat import SenseHat

            self.sense = SenseHat()
            self.has_sense = True

            print("Sense HAT found and initialized.")

        except ImportError:
            self.sense = None
            self.has_sense = False

            print("Sense HAT not found. Running in fallback mode.")

    def start(self):
        self.running = True

        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )

        self.thread.start()

    def _loop(self):
        while self.running:
            if self.has_sense:
                for event in self.sense.stick.get_events():
                    if (
                        event.action == "pressed"
                        and event.direction == "middle"
                    ):
                        print("Hardware Joystick pressed!")
                        self.trigger_callback()

                    elif event.action == "pressed" and event.direction == "left":
                        print("Joystick LEFT pressed")
                        self.trigger_callback("left")

                    elif event.action == "pressed" and event.direction == "right":
                        print("Joystick RIGHT pressed")
                        self.trigger_callback("right")

                    elif event.action == "pressed" and event.direction == "up":
                        print("Joystick UP pressed")
                        self.trigger_callback("up")

            time.sleep(0.1)

    def display_material(self, material):
        """Display material color for 5 seconds."""

        if not self.has_sense:
            return

        color = self.MATERIAL_COLORS.get(material.lower())

        if color is None:
            print(f"Unknown material: {material}")
            return

        with self.led_lock:
            # Invalidate and cancel the previous timer
            self.led_generation += 1
            generation = self.led_generation

            if self.led_timer is not None:
                self.led_timer.cancel()

            try:
                # Show detected material color
                self.sense.clear(*color)

                print(f"Sense HAT color: {material}")

                # Schedule LED shutoff after 5 seconds
                self.led_timer = threading.Timer(
                    5.0,
                    self._clear_leds,
                    args=(generation,)
                )

                self.led_timer.daemon = True
                self.led_timer.start()

            except Exception as error:
                print(f"Sense HAT LED error: {error}")

    def _clear_leds(self, generation):
        """Turn off LEDs if no newer prediction has arrived."""

        with self.led_lock:
            if generation != self.led_generation:
                return

            try:
                self.sense.clear()
                self.led_timer = None
                print("Sense HAT LEDs turned off.")

            except Exception as error:
                print(f"Error clearing LEDs: {error}")

    def stop(self):
        self.running = False

        with self.led_lock:
            self.led_generation += 1

            if self.led_timer is not None:
                self.led_timer.cancel()
                self.led_timer = None

            if self.has_sense:
                self.sense.clear()
import threading
import time


