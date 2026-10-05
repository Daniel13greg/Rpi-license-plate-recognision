"""Presence sensor on a GPIO pin (induction loop detector relay, photo-electric beam, radar...).

Never connect a 12/24 V sensor output straight to the Pi: GPIO pins take 3.3 V at most.
Use the sensor's relay (dry contact) between the pin and GND with ``pull_up: true`` and
``active_high: false``, or an optocoupler module.
"""

from __future__ import annotations

from carwash_lpr.config import TriggerConfig


class GpioSensor:
    def __init__(self, pin: int, active_high: bool = True, pull_up: bool | None = None):
        try:
            from gpiozero import DigitalInputDevice
        except ImportError as exc:
            raise RuntimeError(
                "gpiozero is not available: install python3-gpiozero and create the virtualenv "
                "with --system-site-packages"
            ) from exc
        self.pin = pin
        self.active_high = active_high
        self.pull_up = pull_up
        if pull_up is None:
            self._device = DigitalInputDevice(pin, pull_up=None, active_state=active_high)
        else:
            self._device = DigitalInputDevice(pin, pull_up=pull_up)

    @classmethod
    def from_config(cls, cfg: TriggerConfig) -> "GpioSensor":
        assert cfg.gpio_pin is not None
        return cls(cfg.gpio_pin, cfg.active_high, cfg.pull_up)

    @property
    def active(self) -> bool:
        is_active = bool(self._device.is_active)
        if self.pull_up is None:
            return is_active
        # gpiozero inverts pulled-up inputs (active = low); work back to the pin level
        level_high = (not is_active) if self.pull_up else is_active
        return level_high if self.active_high else not level_high

    def close(self) -> None:
        self._device.close()
