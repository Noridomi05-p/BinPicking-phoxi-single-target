from __future__ import annotations

from copy import deepcopy
import queue
from pathlib import Path
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .contract import build_contract, integer, read_configuration
from .controller import write_controller
from .protocol import RunService
from .reference import BackgroundService, save_camera_profile
from .vision import CAMERA_KEYS, override_settings

LABELS = {
    'OPEN_AT_WAIT': 'Mở kẹp tại điểm chờ', 'ABOVE_TARGET': 'Đến trên vùng gắp',
    'PREGRASP': 'Tiếp cận vật', 'GRASP': 'Hạ đến điểm gắp', 'CLOSE_GRIPPER': 'Đóng và kiểm tra kẹp',
    'RETREAT_TO_PREGRASP': 'Rút ngược hướng tiếp cận', 'LIFT_TO_TRANSFER_Z': 'Nâng qua vành thùng',
    'WAIT_WITH_OBJECT': 'Chuyển về điểm chờ cùng vật', 'DROP_UP_WITH_OBJECT': 'Đến trên băng tải',
    'DROP': 'Hạ đến điểm thả', 'RELEASE_AT_DROP': 'Mở kẹp thả vật',
    'DROP_UP_EMPTY': 'Rút lên khỏi điểm thả', 'WAIT_EMPTY': 'Về P_WAIT — chuẩn bị lượt mới',
}


class ContinuousGUI:
    def __init__(self, root, vision_path, continuous_path):
        self.root, self.vision_path, self.continuous_path = root, Path(vision_path), Path(continuous_path)
        self.events = queue.Queue()
        self.service = self.worker = None
        self.operation = 'RUN'
        self.close_when_finished = False
        self.photo = None
        self.config_var = tk.StringVar(value=str(self.vision_path))
        self.source_var = tk.StringVar(value='phoxicontrol')
        self.count_var = tk.StringVar(value='5')
        self.status = tk.StringVar(value='Chưa chạy — camera đang tự do để chỉnh trong PhoXi Control')
        self.count = tk.StringVar(value='Đã gắp–thả: 0')
        self.summary = tk.StringVar(value='Chọn cấu hình picking_robot.yaml đã dùng khi thử gắp')
        self.target = tk.StringVar(value='Chưa có pose. Mỗi lượt sẽ chụp mới khi robot tại P_WAIT.')
        self.camera_vars = {k: tk.StringVar(value=v) for k, v in
                            dict(texture_source='LED', shutter_multiplier='1', laser_power='2048', led_power='2048').items()}
        self.actual_camera = tk.StringVar(value='Chưa kết nối camera; chưa đọc hay ghi thông số.')
        self.last_output = None
        root.title('PhoXi · Gắp liên tục với Doosan M0609')
        root.geometry('1200x880')
        root.minsize(950, 720)
        self._widgets()
        self._load_defaults()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self._drain)

    def _widgets(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='PhoXi · Gắp liên tục', font=('Sans', 18, 'bold')).pack(anchor='w')
        ttk.Label(outer, textvariable=self.summary, wraplength=1120).pack(anchor='w', pady=(3, 10))
        cfg = ttk.Frame(outer)
        cfg.pack(fill='x')
        self.path_entry = ttk.Entry(cfg, textvariable=self.config_var)
        self.path_entry.pack(side='left', fill='x', expand=True)
        self.choose_button = ttk.Button(cfg, text='Chọn cấu hình pose', command=self.choose_config)
        self.choose_button.pack(side='left', padx=6)
        self.edit_button = ttk.Button(cfg, text='Mở cấu hình chạy', command=lambda: self.open_file(self.continuous_path))
        self.edit_button.pack(side='left')
        tabs = ttk.Notebook(outer)
        tabs.pack(fill='both', expand=True, pady=10)
        run, camera, pendant = ttk.Frame(tabs, padding=10), ttk.Frame(tabs, padding=14), ttk.Frame(tabs, padding=14)
        tabs.add(run, text='Vận hành')
        tabs.add(camera, text='Camera')
        tabs.add(pendant, text='Mã Task Writer')
        control = ttk.Frame(run)
        control.pack(fill='x')
        ttk.Label(control, text='Số lượt tối đa (0: theo MainSub):').pack(side='left')
        self.count_entry = ttk.Spinbox(control, from_=0, to=1000000, textvariable=self.count_var, width=8)
        self.count_entry.pack(side='left', padx=(5, 16))
        self.start_button = ttk.Button(control, text='Bắt đầu chạy', command=self.start)
        self.start_button.pack(side='left')
        self.stop_button = ttk.Button(control, text='Dừng sau lượt hiện tại', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', padx=8)
        ttk.Label(control, textvariable=self.count, font=('Sans', 11, 'bold')).pack(side='right')
        ttk.Label(run, textvariable=self.status, wraplength=1100, font=('Sans', 11, 'bold')).pack(anchor='w', pady=10)
        ttk.Label(run, text='Dừng ngay: dùng Task Stop trên pendant. Nút trên máy tính chỉ ngừng cấp lượt mới.').pack(anchor='w')
        self.preview = ttk.Label(run, text='Ảnh 2D và point cloud Z sẽ hiện sau lần chụp đầu.\nChưa bấm Bắt đầu: chương trình chưa kết nối camera hoặc robot.', anchor='center')
        self.preview.pack(fill='both', expand=True, pady=10)
        ttk.Label(run, textvariable=self.target, wraplength=1100).pack(anchor='w')
        self.log = tk.Text(run, height=7, wrap='word', state='disabled', font=('Monospace', 9))
        self.log.pack(fill='x', pady=(8, 0))
        ttk.Label(camera, text='Nguồn thông số chụp', font=('Sans', 12, 'bold')).grid(row=0, column=0, sticky='w', pady=6)
        self.source_combo = ttk.Combobox(camera, textvariable=self.source_var, values=['phoxicontrol', 'config'], state='readonly', width=22)
        self.source_combo.grid(row=0, column=1, sticky='w')
        ttk.Label(camera, text='phoxicontrol: giữ các thông số đang chỉnh trong PhoXi Control.\nconfig: áp dụng các giá trị dưới đây rồi đọc lại để kiểm tra.', wraplength=900).grid(row=1, column=0, columnspan=3, sticky='w', pady=10)
        self.camera_entries = []
        for row, (key, label) in enumerate([('texture_source', 'Texture source'), ('shutter_multiplier', 'Shutter multiplier'), ('laser_power', 'Laser power'), ('led_power', 'LED power')], 2):
            ttk.Label(camera, text=label).grid(row=row, column=0, sticky='w', pady=5)
            entry = ttk.Entry(camera, textvariable=self.camera_vars[key], width=22)
            entry.grid(row=row, column=1, sticky='w')
            self.camera_entries.append(entry)
        ttk.Separator(camera).grid(row=6, column=0, columnspan=3, sticky='ew', pady=18)
        ttk.Label(camera, text='Đọc lại từ camera khi chạy', font=('Sans', 11, 'bold')).grid(row=7, column=0, columnspan=3, sticky='w')
        ttk.Label(camera, textvariable=self.actual_camera, wraplength=900).grid(row=8, column=0, columnspan=3, sticky='w', pady=10)
        ttk.Label(camera, text='Sau khi dừng và chương trình báo đã nhả camera, bạn có thể kết nối lại trong PhoXi Control.').grid(row=9, column=0, columnspan=3, sticky='w', pady=10)
        ttk.Label(camera, text='Tạo nền mới: dọn hết vật trong thùng, giữ robot tại P_WAIT và không chạy Task Writer.\nChương trình dùng profile đang chọn, kiểm tra mặt phẳng và sao lưu nền cũ trước khi thay.', wraplength=950).grid(row=10, column=0, columnspan=3, sticky='w', pady=(14,6))
        self.background_button = ttk.Button(camera, text='Chụp lại nền — thùng đã trống', command=self.prepare_background)
        self.background_button.grid(row=11, column=0, columnspan=2, sticky='w', pady=6)
        self.save_camera_button = ttk.Button(camera, text='Lưu lựa chọn camera', command=self.save_camera)
        self.save_camera_button.grid(row=11, column=2, sticky='w', padx=12, pady=6)
        self.source_combo.bind('<<ComboboxSelected>>', lambda e: self._camera_controls())
        ttk.Label(pendant, text='Hai khối Custom Code', font=('Sans', 13, 'bold')).pack(anchor='w')
        ttk.Label(pendant, text='1. Khối trước MainSub: toàn bộ mã khởi tạo và định nghĩa hàm.\n2. Khối trong MainSub: chỉ gọi phoxi_run_once().\nĐặt số lần lặp MainSub bằng số lượt tối đa trên giao diện.\nMỗi lần sửa TCP, điểm cố định, tốc độ hoặc Modbus: xuất và dán lại mã khởi tạo.', wraplength=950).pack(anchor='w', pady=12)
        buttons = ttk.Frame(pendant)
        buttons.pack(anchor='w')
        self.export_buttons = []
        for title, part in [('Sao chép mã trước MainSub', 0), ('Sao chép mã trong MainSub', 1), ('Xuất hai tệp mã', None)]:
            button = ttk.Button(buttons, text=title, command=lambda p=part: self.export(p))
            button.pack(side='left', padx=(0, 8))
            self.export_buttons.append(button)
        ttk.Label(pendant, text='TCP và các điểm lấy từ cấu hình hiện tại. Chương trình không tự bật servo hoặc kích hoạt lại kẹp.\nBan đầu cần robot đứng tại P_WAIT, kẹp rỗng, các tín hiệu Modbus đã cấu hình đúng.', wraplength=950).pack(anchor='w', pady=18)
        ttk.Label(outer, text='Chạy theo tuyến cố định đã thử; chưa có lập kế hoạch tránh va chạm toàn tay máy.', foreground='#725323').pack(anchor='w')

    def _load_defaults(self):
        try:
            cfg, policy, contract, watched = read_configuration(self.config_var.get(), self.continuous_path)
            self.count_var.set(str(policy['max_cycles']))
            self.source_var.set(policy['camera_settings_source'])
            for k, v in policy['camera_overrides'].items():
                self.camera_vars[k].set(str(v))
            self._set_summary(contract, policy)
        except Exception as exc:
            self.summary.set('Cần cấu hình trước khi chạy: '+str(exc))
        self._camera_controls()

    def _set_summary(self, contract, policy):
        self.summary.set('TCP: %s · mở 75 mm / PR=%s · đóng PR=%s · TCP Z chuyển vùng=%g mm · %s:%s'%(
            contract['tcp_name'], contract['gripper']['open_pr'], contract['gripper']['close_pr'],
            contract['motion']['transfer_tcp_z_mm'], policy['listen'], policy['port']))

    def settings(self):
        cfg, policy, contract, watched = read_configuration(self.config_var.get(), self.continuous_path)
        policy['max_cycles'] = integer(self.count_var.get(), 0, 1000000, 'MAX_CYCLES')
        policy['camera_settings_source'] = self.source_var.get()
        policy['camera_overrides'] = {k: v.get() for k, v in self.camera_vars.items()}
        if policy['camera_settings_source'] == 'config':
            override_settings(policy['camera_overrides'])
        self._set_summary(contract, policy)
        return cfg, policy, contract, watched

    def choose_config(self):
        path = filedialog.askopenfilename(initialdir=self.vision_path.parent, filetypes=[('YAML', '*.yaml *.yml')])
        if path:
            self.config_var.set(path)
            self._load_defaults()

    def open_file(self, path):
        try:
            subprocess.Popen(['xdg-open', str(Path(path).resolve())])
        except OSError as exc:
            messagebox.showerror('Không mở được tệp', str(exc))

    def export(self, copy_part):
        try:
            cfg, policy, contract, watched = self.settings()
            output = Path(cfg['_project_root'])/'generated'
            paths = write_controller(contract, policy, output)
            if copy_part is not None:
                self.root.clipboard_clear()
                self.root.clipboard_append(paths[copy_part].read_text(encoding='utf-8'))
                self.root.update_idletasks()
                self.status.set('Đã sao chép mã '+('trước MainSub' if copy_part == 0 else 'trong MainSub'))
            else:
                self.status.set('Đã xuất: '+str(output))
            self._append('Mã Task Writer: '+str(paths[0]))
        except Exception as exc:
            messagebox.showerror('Chưa xuất được mã', str(exc))

    def _camera_controls(self):
        state = 'normal' if self.worker is None and self.source_var.get() == 'config' else 'disabled'
        for entry in self.camera_entries:
            entry.configure(state=state)

    def _active(self, active):
        for w in [self.start_button, self.path_entry, self.choose_button, self.edit_button, self.count_entry,
                  self.background_button, self.save_camera_button]+self.export_buttons:
            w.configure(state='disabled' if active else 'normal')
        self.source_combo.configure(state='disabled' if active else 'readonly')
        self.stop_button.configure(state='normal' if active else 'disabled')
        self.stop_button.configure(text='Hủy chụp nền' if self.operation == 'BACKGROUND' else 'Dừng sau lượt hiện tại')
        self._camera_controls()

    def start(self):
        if self.worker is not None:
            return
        try:
            cfg, policy, contract, watched = self.settings()
        except Exception as exc:
            messagebox.showerror('Chưa chạy được', str(exc))
            return
        self.count.set('Đã gắp–thả: 0')
        self.operation = 'RUN'
        self.service = RunService(cfg, policy, contract, watched, emit=lambda kind, value: self.events.put((kind, value)))
        self.worker = threading.Thread(target=self.service.run, name='phoxi-continuous', daemon=True)
        self._active(True)
        self.worker.start()

    def prepare_background(self):
        if self.worker is not None:
            return
        try:
            cfg, policy, contract, watched = self.settings()
        except Exception as exc:
            messagebox.showerror('Chưa chụp được nền', str(exc))
            return
        self.operation = 'BACKGROUND'
        self.service = BackgroundService(cfg, policy, emit=lambda kind,value: self.events.put((kind,value)))
        self.worker = threading.Thread(target=self.service.run, name='phoxi-background', daemon=True)
        self._active(True)
        self.status.set('Đang chuẩn bị chụp nền thùng trống — không kết nối robot')
        self.worker.start()

    def save_camera(self):
        if self.worker is not None:
            return
        try:
            backup = save_camera_profile(self.continuous_path,self.source_var.get(),
                                         {k:v.get() for k,v in self.camera_vars.items()})
            self.status.set('Đã lưu lựa chọn camera cho lần mở tiếp theo')
            self._append('Lưu camera: '+str(self.continuous_path)+' · bản sao: '+str(backup))
        except Exception as exc:
            messagebox.showerror('Chưa lưu được camera',str(exc))

    def stop(self):
        if self.service is not None:
            self.service.request_stop()
            self.status.set('Đang yêu cầu hủy chụp nền; chờ tác vụ kết thúc' if self.operation == 'BACKGROUND'
                            else 'Đã yêu cầu dừng — hoàn tất lượt hiện tại, không cấp lượt mới')
            self.stop_button.configure(state='disabled')

    def close(self):
        if self.worker is not None:
            self.close_when_finished = True
            self.stop()
        else:
            self.root.destroy()

    def _append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text+'\n')
        if int(self.log.index('end-1c').split('.')[0]) > 300:
            self.log.delete('1.0', '100.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def _drain(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == 'STATUS':
                    self.status.set(value)
                elif kind == 'IMAGE':
                    from PIL import Image, ImageTk
                    image = Image.fromarray(value[:, :, ::-1])
                    image.thumbnail((1100, 430))
                    self.photo = ImageTk.PhotoImage(image)
                    self.preview.configure(image=self.photo, text='')
                elif kind == 'CAMERA':
                    values = value.get('values', {})
                    self.actual_camera.set('\n'.join('%s: %s'%(k, values.get(v, 'không đọc được')) for k, v in CAMERA_KEYS.items()))
                    if value.get('read_messages'):
                        self._append('Camera readback: '+str(value['read_messages']))
                elif kind == 'SCAN':
                    p = value['picking'].get('tcp_preview') or {}
                    self.target.set('Frame %s · mục tiêu %s\nGRASP / BASE: %s\nPREGRASP / BASE: %s'%(
                        value['frame_id'], value['picking'].get('target_instance_id'),
                        self._pose_text(p.get('tcp_xyzabc')), self._pose_text(p.get('pregrasp_xyzabc'))))
                elif kind == 'REFERENCE':
                    if value['status'] != 'MATCH':
                        self._append('Nền tham chiếu: '+value['status']+' · '+value['reference_path'])
                        for key, pair in value['differences'].items():
                            self._append('%s: nền=%s; ảnh hiện tại=%s'%(key, pair['saved'], pair['current']))
                        self._append('Giữ bộ thông số đã đọc lại đúng. Nếu nền cũ sai thông tin, tạo lại nền khi thùng trống; không bỏ kiểm tra nền.')
                elif kind == 'EVENT':
                    self.count.set('Đã gắp–thả: '+str(value['completed']))
                    step = value.get('step')
                    if step:
                        self.status.set(LABELS.get(step, step))
                    self._append(value['kind']+': '+str(step or value.get('reason') or value.get('detail') or value.get('reasons') or value['phase']))
                    if value['kind'] in ('arm_received','capture_expired_before_go'):
                        self._append('Tuổi ảnh %.0f / %.0f ms; chờ xác nhận ARM %.0f ms'%(
                            value['capture_age_ms'],value['max_frame_age_ms'],value['arm_roundtrip_ms']))
                elif kind == 'FINISHED':
                    self.last_output = value['output']
                    self.status.set('Đã dừng tại điểm chờ: '+value['reason'])
                    self.count.set('Đã gắp–thả: '+str(value['completed']))
                    self._append('Log: '+value['output'])
                elif kind == 'BACKGROUND_SAVED':
                    self.status.set('Đã lưu nền mới đúng profile — đưa vật vào thùng rồi Bắt đầu chạy')
                    self._append('Nền mới: '+value['path']+' · số ảnh='+str(value['frames']))
                    self._append('Độ phủ nền trong ROI: %.1f%%'%(100*value['coverage']))
                    if value.get('backup'):
                        self._append('Bản sao nền cũ: '+value['backup'])
                elif kind == 'FAULT':
                    self.last_output = value['output']
                    self.status.set('Đã ngừng cấp lệnh: '+value['detail'])
                    self._append('LỖI tại '+value['phase']+': '+value['detail'])
                    if value['detail'].startswith('REFERENCE_IDENTITY_UNVERIFIED'):
                        self.status.set('Nền chưa khớp thông số ảnh — xem từng trường lệch trong log; giữ profile camera đúng và tạo lại nền khi thùng trống')
                    elif value['detail'] == 'CAPTURE_EXPIRED_BEFORE_GO':
                        self.status.set('Ảnh quá hạn trước GO — chưa gửi lệnh chuyển động; xem tuổi ảnh và thời gian chờ ARM trong log')
                    if value['motion_outcome_unconfirmed']:
                        self._append('PC chưa xác nhận kết quả chuyển động. Kiểm tra robot/Task Stop trước khi chạy lại.')
                elif kind == 'RELEASED':
                    self._append(value)
                    self.worker = self.service = None
                    self.operation = 'RUN'
                    self._active(False)
                    if self.close_when_finished:
                        self.root.destroy()
                        return
                else:
                    self._append(kind+': '+str(value))
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    @staticmethod
    def _pose_text(values):
        return '—' if not values else '['+', '.join('%.2f'%v for v in values)+']'
