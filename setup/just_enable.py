import openarm_can as oa
import time
# Create OpenArm instance

right_arm = oa.OpenArm("can2", True)
left_arm = oa.OpenArm("can3", True)

# Initialize arm motors
motor_types = [
    oa.MotorType.DM8009,
    oa.MotorType.DM8009,
    oa.MotorType.DM4340,
    oa.MotorType.DM4340,
    oa.MotorType.DM4310,
    oa.MotorType.DM4310,
    oa.MotorType.DM4310
]
send_ids = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
recv_ids = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
right_arm.init_arm_motors(motor_types, send_ids, recv_ids)
left_arm.init_arm_motors(motor_types, send_ids, recv_ids)

# Initialize gripper
right_arm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
left_arm.init_gripper_motor(oa.MotorType.DM4310, 0x08, 0x18)
right_arm.set_callback_mode_all(oa.CallbackMode.IGNORE)
left_arm.set_callback_mode_all(oa.CallbackMode.IGNORE)
# Use high-level operations
right_arm.enable_all()
right_arm.recv_all(2500)
left_arm.enable_all()
left_arm.recv_all(2500)


# return to zero position
right_arm.set_callback_mode_all(oa.CallbackMode.STATE)
right_arm.get_arm().mit_control_all([oa.MITParam(0.0, 0.0, 0.0, 0.0, 0.0),
                                     oa.MITParam(0.0, 0.0, 0.0, 0.0, 0.0)])

right_arm.recv_all(2500)

try:
    while True:
        right_arm.refresh_all()
        right_arm.recv_all()
        for motor in right_arm.get_arm().get_motors():
            print(motor.get_position())
        for motor in right_arm.get_gripper().get_motors():
            print(motor.get_position())
except KeyboardInterrupt:
    pass

right_arm.disable_all()
left_arm.disable_all()
right_arm.recv_all()
left_arm.recv_all()