# Hardware Module — Placeholder

This directory is reserved for future physical hardware integrations, such as:

- GPIO control (Raspberry Pi LEDs, buttons, servos)
- Display drivers (SSD1306 OLED, small TFT screens)
- Sensor input (distance, temperature, motion)
- Robot body actuation

All hardware modules should subscribe to the event bus (`core/event_bus`)
so they react to pipeline state without coupling to voice logic directly.
