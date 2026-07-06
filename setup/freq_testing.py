import openarm_can as oa
import time
# Create OpenArm instance

arm = oa.OpenArm("can1", True)

# Initialize arm motors
motor_types = [
    oa.MotorType.DM8009,
    oa.MotorType.DM8009,
    oa.MotorType.DM4340,
    oa.MotorType.DM4340,
    oa.MotorType.DM4310,
    oa.MotorType.DM4310,
    oa.MotorType.DM4310,
    oa.MotorType.DM4310]

send_ids = [ 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 ]
recv_ids = [ 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18 ]
arm.init_arm_motors(motor_types, send_ids, recv_ids)

arm.set_callback_mode_all(oa.CallbackMode.STATE)

import time

n = 0
end_time = time.time() + 10.0  # Run for 10 seconds

while time.time() < end_time:
    arm.disable_all()
    arm.recv_all(610)
    n += 1

print(f"Frequency: {n / 10.0} Hz")