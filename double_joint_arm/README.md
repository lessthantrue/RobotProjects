# Double jointed arm

2 joint arm that reads G code to draw to the screen

Features:

* Simulation of motor and arm dynamics (robot/motor_dynamics.py, robot/robot_state_full.py)
* Generalized motion profile generator (profile/profile.py)
* Inverse kinematics and PID controller path following (profile/profile.py, controllers/velocity_regulator.py)
* G code interpreter supporting G0, G1, G2, G3 (interpreter.py)
* Dynamics bounded path generation (paths/*.py)
