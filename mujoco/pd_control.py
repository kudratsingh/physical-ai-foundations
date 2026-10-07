"""Week 1 (Friday): close the loop on one_joint.xml with a PD controller.

    error = q_target - q
    u     = Kp * error - Kd * qvel
    data.ctrl[0] = u          # motor actuator, gear=1 -> u is a joint torque

TODO: run at least three (Kp, Kd) settings, including one that oscillates or
converges badly, plot target vs actual angle, and write 5-10 sentences on what
changed. Reuse load_model() / reset() from simulate.py.
"""
