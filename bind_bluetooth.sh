#!/bin/bash

# MAC address of the device
MAC_ADDRESS="3C:8A:1F:D3:A3:3A"

echo "Starting Bluetooth binding process..."

# Enable Bluetooth and agent
echo "Configuring Bluetooth..."
bluetoothctl << EOF
power on
agent on
default-agent
EOF

# Try to connect and trust the device
echo "Attempting to pair and trust device..."
bluetoothctl << EOF
pair $MAC_ADDRESS
trust $MAC_ADDRESS
connect $MAC_ADDRESS
exit
EOF

# Release existing bindings if any
echo "Releasing any existing RFCOMM bindings..."
sudo rfcomm release 0 2>/dev/null

# Bind the device to rfcomm0
echo "Binding device to /dev/rfcomm0..."
sudo rfcomm bind 0 $MAC_ADDRESS

# Check if binding was successful
if [ -e /dev/rfcomm0 ]; then
    echo "Success! Device bound to /dev/rfcomm0"
    echo "You can now run your Python script"
else
    echo "Error: Failed to create /dev/rfcomm0"
    exit 1
fi