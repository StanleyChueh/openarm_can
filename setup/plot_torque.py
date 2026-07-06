import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_torque_analysis(csv_filename):
    try:
        df = pd.read_csv(csv_filename)
        # 列印出所有找到的欄位，方便除錯
        print(f"成功讀取檔案。找到的欄位有: {list(df.columns)}")
    except FileNotFoundError:
        print(f"找不到檔案: {csv_filename}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('OpenArm Torque Performance Analysis', fontsize=16)

    # --- 第一張圖：左手 ID 4 (固定顯示) ---
    if 'L_J4_cmd' in df.columns:
        ax1.plot(df['time'].values, df['L_J4_cmd'].values, label='L_J4 Command', color='blue', linestyle='--')
        ax1.plot(df['time'].values, df['L_J4_act'].values, label='L_J4 Actual', color='cyan', alpha=0.7)
        ax1.set_title('Left Arm ID 4 (Elbow) Torque')
    else:
        print("警告: 找不到 L_J4 相關欄位")

    ax1.set_ylabel('Torque (Nm)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 檢查是左手還是右手的 ID 1
    if 'L_J1_cmd' in df.columns:
        col_cmd, col_act, label_name, color_theme = 'L_J1_cmd', 'L_J1_act', 'Left J1', 'green'
    elif 'R_J1_cmd' in df.columns:
        col_cmd, col_act, label_name, color_theme = 'R_J1_cmd', 'R_J1_act', 'Right J1 (Old Log)', 'red'
    else:
        print("錯誤: 找不到任何 ID 1 (L_J1 或 R_J1) 的數據")
        plt.close()
        return

    ax2.plot(df['time'].values, df[col_cmd].values, label=f'{label_name} Cmd', color=color_theme, linestyle='--')
    ax2.plot(df['time'].values, df[col_act].values, label=f'{label_name} Act', color='lime' if color_theme=='green' else 'orange', alpha=0.7)
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_title(f'{label_name} Torque Analysis')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('latest_torque_analysis.png', dpi=300)
    print("圖表已儲存為 latest_torque_analysis.png")
    plt.show()

if __name__ == "__main__":
    plot_torque_analysis("torque_startup_test.csv")