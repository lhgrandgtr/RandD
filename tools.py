import serial
import time
import logging
from semantic_kernel.functions import kernel_function


class RemoteController:
    def __init__(self, bluetooth_port, baud_rate, timeout=1):
        logging.info("Initializing RemoteController ...")
        self.bluetooth_port = bluetooth_port
        self.baud_rate = baud_rate
        try:
            self.bt = serial.Serial(port=bluetooth_port, baudrate=baud_rate, timeout=timeout)
            logging.info(f"Connected to {bluetooth_port} at {baud_rate} baud.")
        except serial.SerialException as e:
            logging.error(f"Could not connect to Bluetooth device: {e}")
            self.bt = None
        

    def send_command(self, command):

        if self.bt is None:
            logging.error("Bluetooth connection not established.")
            return
        self.bt.write(command.encode())
        print(f"Sent command: {command}")


    @kernel_function(
        name="move_forward",
        description="Move the car forward for a specified time period",
    )
    async def forward(self, period=1):
        """Move forward for a specified period."""
        logging.info("Moving forward")
        while period > 0:
            self.send_command('U')
            time.sleep(0.1)
            period -= 0.1
        logging.info("Stopped moving forward")
        return 'Stopped moving forward'



    @kernel_function(
        name="move_backward",
        description="Move the car backward for a specified time period",
    )
    async def backward(self, period=1):
        """Move backward for a specified period."""
        logging.info("Moving backward")
        while period > 0:
            self.send_command('D')
            time.sleep(0.1)
            period -= 0.1
        logging.info("Stopped moving backward")
        return 'Stopped moving backward'


    @kernel_function(
        name="turn_left",
        description="Turn the car left for a specified time period",
    )
    async def left(self, period=1):
        """Turn left for a specified period."""
        logging.info("Turning left")
        while period > 0:
            self.send_command('L')
            time.sleep(0.1)
            period -= 0.1
        logging.info("Stopped turning left")
        return 'Stopped turning left'


    @kernel_function(
        name="turn_right", 
        description="Turn the car right for a specified time period",
    )
    def right(self, period=1):
        """Turn right for a specified period."""
        logging.info("Turning right")
        while period > 0:
            self.send_command('R')
            time.sleep(0.1)
            period -= 0.1
        logging.info("Stopped turning right")
        return 'Stopped turning right'


    @kernel_function(
        name="stop_car",
        description="Stop all car movement immediately",
    )
    def stop(self):
        """Stop the car."""
        logging.info("Stopping")
        self.send_command('X')
        logging.info("Stopped")
        return 'Stopped'


    @kernel_function(
        name="set_car_speed",
        description="Set the speed of the car (value between 0-9)",
    )
    def set_speed(self, speed):
        """Set the speed of the car."""
        if speed < 0 or speed > 9:
            logging.error("Speed must be an integer between 0 and 9.")
            return
        logging.info(f"Setting speed to {speed}")
        self.send_command(str(speed))
        logging.info("Speed set")
        return f'Speed set to {speed}'


if __name__ == "__main__":
    controller = RemoteController()
