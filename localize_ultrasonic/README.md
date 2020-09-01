# Localization with Ultrasonic Sensors

A team of robots ahat move around a room in a rigid shape, estimating their position with ultrasonic sensors

Method based on [Multi-Ray Modeling of Ultrasonic Sensors and Application for Micro-UAV Localization in Indoor Environments](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6514676/#FD3-sensors-19-01770)

Features:

* Extended kalman filter for single robot localization (estimator.py)
* Holonomic motion robot model (motion.py)
* Keyboard controls for testing (key_input.py)
* PIDF controller for coordinated team movement (controller.py, regulator.py)
* Ultrasonic sensors modeled in accordance with paper mentioned above (sensor.py)
* randomly generated world (world.py)
