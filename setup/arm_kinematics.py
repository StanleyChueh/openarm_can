#arm_kinematics.py
import pinocchio as pin
import numpy as np

from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

class ArmKinematics:
    def __init__(self, urdf_path: str, end_effector_name: str):
        """
        Initialize arm kinematics for dual-arm system.
        
        Args:
            urdf_path: Path to URDF file
            end_effector_name: Name of end-effector frame (e.g., 'openarm_left_hand_tcp')
        """
        # 建立模型與資料
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        
        self.ee_id = self.model.getFrameId(end_effector_name)
        
        self.q = np.zeros(self.model.nq, dtype=np.float32)
        self.dq = np.zeros(self.model.nv, dtype=np.float32)
        
        self.kv_indices = np.zeros(self.model.nq, dtype=np.float32)
        self.kc_indices = np.zeros(self.model.nq, dtype=np.float32)

        kv_left = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        kc_left = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        kv_right = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        kc_right = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.kv_indices[0:7] = kv_left
        self.kc_indices[0:7] = kc_left
        
        self.kv_indices[9:16] = kv_right
        self.kc_indices[9:16] = kc_right
        
        self.v_max = 0.3
        
        self.q0 = np.zeros(self.model.nq, dtype=np.float32)
        
        self.friction_smoothness = 50.0
    
    def solve_pos(self, q: list[float]) -> NDArray[np.float32]:
        pin.forwardKinematics(self.model, self.data, np.array(q, dtype=np.float32))
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_id].translation.copy()

    def get_gripper_rotation(self, q: list[float]) -> NDArray[np.float32]:
        pin.forwardKinematics(self.model, self.data, np.array(q, dtype=np.float32))
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.ee_id].rotation.copy()

    def solve_ik(self, q0: NDArray[np.float32], p_des: NDArray[np.float32], gripper=0.2, R_des: None | NDArray[np.float32]=None,
        alpha=1.0, lambda_=0.1, max_iter=1000, tol=1e-3):
        
        q = q0.copy()

        for _ in range(max_iter):
            # Forward Kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            p = self.data.oMf[self.ee_id].translation

            # 誤差
            e = p_des - p
            if np.linalg.norm(e) < tol:
                break

            # 雅可比矩陣
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]

            # LM 更新
            H = J.T @ J + lambda_**2 * np.eye(self.model.nq)
            dq = np.linalg.solve(H, J.T @ e)

            q += alpha * dq
        else:
            raise RuntimeError('IK 解失敗，無法達到理想姿態')

        if R_des is None:
            q[-1] = gripper
            return q

        oMdes = pin.SE3(R_des, p_des)
        for _ in range(max_iter):
            # Forward Kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.ee_id]

            # --- 位姿誤差（6 維：位置 + 姿態） ---
            err = pin.log6(oMdes.inverse() * oMf).vector  # [vx, vy, vz, wx, wy, wz]
            if np.linalg.norm(err) < tol:
                break

            # --- 雅可比矩陣（6xN） ---
            J6 = pin.computeFrameJacobian(self.model, self.data, q, self.ee_id, pin.LOCAL)
            H = J6.T @ J6 + (lambda_ ** 2) * np.eye(self.model.nq)
            dq = np.linalg.solve(H, -J6.T @ err)

            q += alpha * dq
        else:
            raise RuntimeError('IK 解失敗，無法達到理想姿態')

        q[-1] = gripper
        return q

    def compute_friction(self, dq: NDArray[np.float32]) -> NDArray[np.float32]:
        # 黏滯摩擦 (Viscous)
        tau_viscous = self.kv_indices * dq
        
        # 庫倫摩擦 (Coulomb) - 使用 tanh 平滑化避免震盪
        tau_coulomb = self.kc_indices * np.tanh(self.friction_smoothness * dq)
        
        return -(tau_viscous + tau_coulomb)
    
    def compute_gravity(self, q: NDArray[np.float32]) -> NDArray[np.float32]:
        tau = pin.computeGeneralizedGravity(self.model, self.data, q)
        
        return tau
    
    

    def compute_time(self, q: NDArray[np.float32], q_des: NDArray[np.float32]):
        dq = np.abs(q_des - q)
        max_dq = np.max(dq)
        T = max_dq / self.v_max
        
        self.q0 = q.copy()
        
        return T
    
    def min_jerk(self, q, q_des, T, t):
        s = np.clip(t / T, 0, 1)
        s3, s4, s5 = s**3, s**4, s**5
        pos = q + (q_des - q) * (10*s3 - 15*s4 + 6*s5)
        vel = (q_des - q) * (30*s**2 - 60*s3 + 30*s4) / T
        acc = (q_des - q) * (60*s - 180*s**2 + 120*s3) / (T**2)
        
        return pos, vel, acc

    def compute_tau(self, q, dq, q_des, t, T, with_ref=False):
        q_ref, dq_ref, ddq_ref = self.min_jerk(self.q0, q_des, T, t)
        
        Kp = np.diag([  6.0,  6.0,   6.0,   6.0,    6.0,    6.0,    6.0])
        Kd = np.diag([  0.4,  0.2,   0.2,   0.2,    0.2,    0.2,    0.2])
        tau = Kp @ (q_ref - q) + Kd @ (dq_ref - dq)
        
        if not with_ref:
            return tau
        else:
            return tau, q_ref, dq_ref