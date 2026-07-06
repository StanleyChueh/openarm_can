from scipy.spatial.transform import Rotation as R
from arm_kinematics import ArmKinematics
import openarm_can as oa
import time
import numpy as np
import sys
import mujoco
import mujoco.viewer

import csv #畫圖


URDF_PATH = 'model/openarm_description.urdf'

MOTOR_TYPES_ARM = [
    oa.MotorType.DM8009, 
    oa.MotorType.DM8009, 
    oa.MotorType.DM4340, 
    oa.MotorType.DM4340, 
    oa.MotorType.DM4310, 
    oa.MotorType.DM4310, 
    oa.MotorType.DM4310, 
    oa.MotorType.DM4310  
]

IDS_SEND = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08] 
IDS_RECV = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]

GEAR_RATIO_DM4340 = 12.5


def motor_pos_to_joint_angle(motor_pos: float, motor_type: oa.MotorType) -> float:
    # if motor_type == oa.MotorType.DM4340:
    #     return (motor_pos / GEAR_RATIO_DM4340) * np.pi
    # else:
    return motor_pos

def motor_vel_to_joint_velocity(motor_vel: float, motor_type: oa.MotorType) -> float:
    # if motor_type == oa.MotorType.DM4340:
    #     return (motor_vel / GEAR_RATIO_DM4340) * np.pi
    # else:
    return motor_vel

def get_arm_data(arm_component, motor_types: list):
    motors = arm_component.get_motors()
    q_list = []
    dq_list = []
    
    for i, motor in enumerate(motors):
        # Position
        raw_pos = motor.get_position()
        q_list.append(motor_pos_to_joint_angle(raw_pos, motor_types[i]))
        # Velocity
        raw_vel = motor.get_velocity()
        dq_list.append(motor_vel_to_joint_velocity(raw_vel, motor_types[i]))
    
    return np.array(q_list, dtype=np.float32), np.array(dq_list, dtype=np.float32)

def merge_dual_arm_state(q_left_8: np.ndarray, q_right_8: np.ndarray, total_dof: int = 18) -> np.ndarray:

    state_full = np.zeros(total_dof, dtype=np.float32)
    
    state_full[0:7] = q_left_8[0:7]
    

    # state_full[9:16] = q_right_8[0:7] * np.array([1, 1, 1, 1, 1, 1, 1])
    
    state_full[9:16] = q_right_8[0:7]
        
    return state_full

def create_mit_params(joint_angles: np.ndarray, kp_values, kd_values, tau_values):
    """建立 MIT 控制參數列表"""
    params = []
    n_joints = len(joint_angles)
    
    # 確保輸入長度一致
    if len(kp_values) != n_joints or len(kd_values) != n_joints or len(tau_values) != n_joints:
        print(f"[ERROR] Params length mismatch! Joints: {n_joints}")
        return []

    for i in range(n_joints):
        params.append(oa.MITParam(
            kp_values[i], 
            kd_values[i], 
            joint_angles[i], 
            0.0, 
            tau_values[i]
        ))
    return params

# ============================================================================
# 3. 主程式初始化 (INITIALIZATION)
# ============================================================================

def main():
    model = mujoco.MjModel.from_xml_path('model/openarm_description.xml') 
    data = mujoco.MjData(model)
    print("[INFO] ========================================")
    print("[INFO] Initializing DUAL ARM Gravity Compensation")
    print("[INFO] ========================================")

    print("[INFO] Connecting to LEFT ARM (can0)...")
    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        left_sys = oa.OpenArm("can1", True)
        left_sys.init_arm_motors(MOTOR_TYPES_ARM, IDS_SEND, IDS_RECV)
        left_sys.enable_all()
        left_sys.recv_all() 
        left_sys.set_callback_mode_all(oa.CallbackMode.STATE) 
        left_arm = left_sys.get_arm()
    except Exception as e:
        print(f"[ERROR] Left arm init failed: {e}")
        return

    print("[INFO] Connecting to RIGHT ARM (can1)...")
    try:
        right_sys = oa.OpenArm("can0", True)
        right_sys.init_arm_motors(MOTOR_TYPES_ARM, IDS_SEND, IDS_RECV)
        right_sys.enable_all()
        right_sys.recv_all()
        right_sys.set_callback_mode_all(oa.CallbackMode.STATE)
        right_arm = right_sys.get_arm()
    except Exception as e:
        print(f"[ERROR] Right arm init failed: {e}")
        left_sys.disable_all()
        return

    print(f"[INFO] Loading URDF: {URDF_PATH}")
    try:
        kinematics = ArmKinematics(URDF_PATH, 'openarm_left_hand_tcp')
        print(f"[INFO] Model DOF: {kinematics.model.nq}")
        # print("Model Gravity Vector:", kinematics.model.gravity.linear)
    except Exception as e:
        print(f"[ERROR] Kinematics init failed: {e}")
        left_sys.disable_all()
        right_sys.disable_all()
        return

    # ============================================================================
    # 4. 控制迴圈 (CONTROL LOOP)
    # ============================================================================
    print("\n[READY] System Ready.")
    print("[WARNING] Gravity compensation starting in 3 seconds...")
    print("[WARNING] PLEASE HOLD BOTH ARMS!")

    DAMPING_GAIN = 0.5
    gravity_gains = [0.0] * 18
    
    gravity_gains[0:7] = [0.85, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4]
    gravity_gains[9:16] = [0.85, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4]
    
    kp_zeros = [0.0] * 8

    kd_damping = [DAMPING_GAIN] * 8
    
    log_filename = "torque_startup_test.csv"
    csv_file = open(log_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['time', 'L_J4_cmd', 'L_J4_act', 'L_J1_cmd', 'L_J1_act'])
    
    try:
        loop_count = 0
        start_time = time.time()
        
        while viewer.is_running():

            left_sys.recv_all()

            right_sys.recv_all()

            

            q_l, dq_l = get_arm_data(left_arm, MOTOR_TYPES_ARM)

            q_r, dq_r = get_arm_data(right_arm, MOTOR_TYPES_ARM) 

            

            q_full = merge_dual_arm_state(q_l, q_r, kinematics.model.nq) 

            dq_full = merge_dual_arm_state(dq_l, dq_r, kinematics.model.nq)

            

            data.qpos[:] = q_full  

            data.qvel[:] = 0    

            mujoco.mj_forward(model, data)

            viewer.sync()        


            tau_gravity = kinematics.compute_gravity(q_full)

            tau_friction = kinematics.compute_friction(dq_full)

            

            tau_total = (tau_gravity * gravity_gains) + tau_friction

            

            tau_l_7 = tau_total[0:7]

            tau_l_cmd = np.append(tau_l_7, 0.0) 

            

            tau_r_7 = tau_total[9:16]

            tau_r_cmd = np.append(tau_r_7, 0.0)

            

            elapsed = time.time() - start_time

            l_j4_act = left_arm.get_motors()[3].get_torque()

            l_j1_act = left_arm.get_motors()[0].get_torque()

            csv_writer.writerow([elapsed, tau_l_cmd[3], l_j4_act, tau_l_cmd[0], l_j1_act])


            params_l = create_mit_params(q_l, kp_zeros, kd_damping, tau_l_cmd)

            left_arm.mit_control_all(params_l)

            

            params_r = create_mit_params(q_r, kp_zeros, kd_damping, tau_r_cmd)

            right_arm.mit_control_all(params_r)

            

            loop_count += 1

            if loop_count % 200 == 0:

                elapsed = time.time() - start_time

                freq = loop_count / elapsed

                print(f"\r[RUNNING] Freq: {freq:.1f} Hz | L_J1_Tau: {tau_l_cmd[0]:.2f} | R_J1_Tau: {tau_r_cmd[0]:.2f}", end="")

            

            time.sleep(0.001)



    except KeyboardInterrupt:

        print("\n\n[INFO] Interrupted by user (Ctrl+C).")

        

    except Exception as e:

        print(f"\n[ERROR] Runtime error: {e}")

        import traceback

        traceback.print_exc()

        

    finally:

        viewer.close()

        csv_file.close()

        print(f"[INFO] Data saved to {log_filename}")

        print("\n[INFO] Disabling all motors...")

        try:

            left_sys.disable_all()

            right_sys.disable_all()

            # 確保命令送出

            left_sys.recv_all()

            right_sys.recv_all()

            print("[INFO] Safety shutdown complete.")

        except:

            print("[ERROR] Error during shutdown.")



if __name__ == "__main__":

    main()