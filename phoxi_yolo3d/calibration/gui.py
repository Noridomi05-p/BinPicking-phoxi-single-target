"""Tk/ttk desktop workflow. Acquisition, fitting and disk IO run outside Tk's thread."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from ..storage import save_capture
from .board import BoardSpec, estimate_board, overlay
from .math3d import parse_pose
from .session import Session, atomic_json
from .solver import solve
from .sources import DemoSource, PhoXiSource


class CalibrationApp:
    def __init__(self, root, config, options, demo=False, session_path=None):
        self.root, self.config, self.options, self.demo = root, config, options, bool(demo)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="calibration-worker")
        self.busy = False
        self.session = None
        self.source = None
        self.frame = self.detection = None
        self.preview_image = self.tk_image = None
        self.reviewing = False
        self.buttons = []
        self.locked_widgets = []
        self.root.title("PhoXi · Hiệu chuẩn với Doosan M0609" + (" · DEMO" if demo else ""))
        self.root.geometry("1180x820")
        self.root.minsize(1000, 700)
        self.root.configure(background="#f3f5f6")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f3f5f6")
        style.configure("TLabel", background="#f3f5f6", foreground="#23313c")
        style.configure("TLabelframe", background="#f3f5f6")
        style.configure("TLabelframe.Label", background="#f3f5f6", foreground="#23313c")
        style.configure("Title.TLabel", font=("DejaVu Sans", 15, "bold"))
        style.configure("Accent.TButton", foreground="#ffffff", background="#176655", padding=7)
        style.map("Accent.TButton", background=[("active", "#215a4f"), ("disabled", "#8b9b98")])
        self.status = tk.StringVar(value="Tạo hoặc mở một phiên hiệu chuẩn để bắt đầu.")
        self.frame_status = tk.StringVar(value="Chưa chụp ảnh")
        self.pair_status = tk.StringVar(value="Chưa có cặp ảnh / pose mới.")
        self.count_text = tk.StringVar(value="0 mẫu tính · 0 mẫu kiểm chứng")
        self.result_text = tk.StringVar(value="Chưa có kết quả")
        self.validation_text = tk.StringVar(value="Chưa có kiểm chứng độc lập")
        self.cols = tk.StringVar(value=str(options.get("board", {}).get("cols", 15)))
        self.rows = tk.StringVar(value=str(options.get("board", {}).get("rows", 10)))
        self.square = tk.StringVar(value=str(options.get("board", {}).get("square_mm", 16)))
        self.device = tk.StringVar(value=str(config["camera"].get("device_id") or ""))
        self.session_name = tk.StringVar(value=self._new_name())
        self.output_dir = tk.StringVar(value=str(Path(config["_project_root"]) / "data/calibration/sessions"))
        self.pos_limit = tk.StringVar(value=str(options.get("solver", {}).get("max_position_error_mm", 2.0)))
        self.rot_limit = tk.StringVar(value=str(options.get("solver", {}).get("max_rotation_error_deg", 1.0)))
        self.pose_vars = [tk.StringVar() for _ in range(6)]
        self.paste_pose = tk.StringVar()
        self.stationary, self.origin_ok, self.flipped = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
        self.role = tk.StringVar(value="Tính calibration")
        self.show_corners = tk.BooleanVar(value=True)
        self._build()
        self.root.bind("<Control-space>", lambda e: self.capture())
        self.root.bind("<Control-Return>", lambda e: self.save_pair())
        for value in self.pose_vars:
            value.trace_add("write", lambda *_: self._refresh_buttons())
        if session_path:
            self._load_session(session_path)
        self._refresh_buttons()

    @staticmethod
    def _new_name():
        return "phoxi_m0609_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    def _button(self, parent, text, command, primary=False, **pack):
        button = ttk.Button(parent, text=text, command=command, style="Accent.TButton" if primary else "TButton")
        button.pack(**pack)
        self.buttons.append(button)
        return button

    def _entry(self, parent, label, var, row, column=0, width=20, locked=False):
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=5, pady=(5,2))
        widget = ttk.Entry(parent, textvariable=var, width=width)
        widget.grid(row=row+1, column=column, sticky="ew", padx=5, pady=(0,5))
        if locked:
            self.locked_widgets.append(widget)
        return widget

    def _build(self):
        header = ttk.Frame(self.root, padding=(20,12))
        header.pack(fill="x")
        ttk.Label(header, text="PhoXi · Hiệu chuẩn với robot", style="Title.TLabel").pack(anchor="w")
        self.mode_label = ttk.Label(header, text=("DEMO · dữ liệu tổng hợp, không dùng điều khiển robot" if self.demo
                                  else "Camera cố định · pose mặt bích / BASE · mm / độ · Euler ZYZ"))
        self.mode_label.pack(anchor="w", pady=(4,0))
        self.book = ttk.Notebook(self.root)
        self.book.pack(fill="both", expand=True, padx=14)
        self.tabs = []
        for name in ("1. Thiết lập", "2. Thu mẫu", "3. Tính kết quả", "4. Kiểm chứng"):
            tab = ttk.Frame(self.book, padding=12)
            self.book.add(tab, text=name)
            self.tabs.append(tab)
        self._setup_tab()
        self._capture_tab()
        self._results_tab()
        self._validation_tab()
        footer = ttk.Frame(self.root, padding=(16,8))
        footer.pack(fill="x")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=110)
        self.progress.pack(side="right", padx=8)
        ttk.Label(footer, textvariable=self.status, wraplength=870).pack(side="left", fill="x", expand=True)

    def _setup_tab(self):
        tab = self.tabs[0]
        tab.columnconfigure((0,1), weight=1)
        camera = ttk.LabelFrame(tab, text="Camera và checkerboard", padding=12)
        camera.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        camera.columnconfigure((0,1,2),weight=1)
        self._entry(camera,"Camera ID",self.device,0,locked=True)
        self._entry(camera,"Cột GÓC TRONG",self.cols,2,0,10,True)
        self._entry(camera,"Hàng GÓC TRONG",self.rows,2,1,10,True)
        self._entry(camera,"Cạnh ô (mm)",self.square,2,2,10,True)
        ttk.Label(camera,text="Ví dụ: bảng 16 × 11 ô → 15 × 10 góc trong.\nĐo kích thước bản in thực tế; dán bảng lên tấm phẳng cứng.",wraplength=450).grid(row=4,column=0,columnspan=3,sticky="w",padx=5,pady=12)
        ttk.Label(camera,text="Ảnh toàn khung; không dùng ROI của chương trình picking.\nGiữ CameraSpace/PrimaryCamera; không ghi ma trận vào scanner.",wraplength=450).grid(row=5,column=0,columnspan=3,sticky="w",padx=5,pady=8)
        right = ttk.LabelFrame(tab,text="Phiên làm việc",padding=12)
        right.grid(row=0,column=1,sticky="nsew",padx=(8,0))
        right.columnconfigure((0,1),weight=1)
        self._entry(right,"Tên phiên",self.session_name,0,locked=True)
        self._entry(right,"Thư mục gốc lưu phiên",self.output_dir,2,locked=True)
        self._entry(right,"Ngưỡng vị trí lớn nhất (mm)",self.pos_limit,4,0,12,True)
        self._entry(right,"Ngưỡng góc lớn nhất (độ)",self.rot_limit,4,1,12,True)
        ttk.Label(right,text="Ngưỡng do người dùng đặt, không phải cam kết độ chính xác.\nĐề xuất 15–20 mẫu tính và ≥5 mẫu kiểm chứng riêng.",wraplength=440).grid(row=6,column=0,columnspan=2,sticky="w",padx=5,pady=12)
        actions = ttk.Frame(tab,padding=(0,20))
        actions.grid(row=1,column=0,columnspan=2,sticky="ew")
        self.new_button = self._button(actions,"Tạo phiên + sang thu mẫu",self.new_session,True,side="left",padx=5)
        self.open_button = self._button(actions,"Mở lại phiên",self.open_session,side="left",padx=5)
        self.unlock_button = self._button(actions,"Chuẩn bị phiên mới",self.prepare_new,side="left",padx=5)
        self.session_label = ttk.Label(tab,text="Chưa mở phiên",wraplength=1000)
        self.session_label.grid(row=2,column=0,columnspan=2,sticky="w",padx=5,pady=10)
        note = "Đánh dấu một GÓC TRONG làm gốc vật lý trên bảng. Gốc và hai trục phải giống nhau ở mọi ảnh.\nBảng gá cứng vào cụm robot; không yêu cầu trùng TCP gắp. Chỉ ghi pose mặt bích trong BASE."
        ttk.Label(tab,text=note,wraplength=1050).grid(row=3,column=0,columnspan=2,sticky="w",padx=5,pady=10)

    def _capture_tab(self):
        tab = self.tabs[1]
        pane = ttk.Panedwindow(tab,orient="vertical")
        pane.pack(fill="both",expand=True)
        top = ttk.Frame(pane)
        pane.add(top,weight=3)
        top.columnconfigure(0,weight=3)
        top.columnconfigure(1,weight=2)
        top.rowconfigure(0,weight=1)
        left=ttk.LabelFrame(top,text="Ảnh texture và góc bảng",padding=8)
        left.grid(row=0,column=0,sticky="nsew",padx=(0,8))
        left.rowconfigure(1,weight=1)
        left.columnconfigure(0,weight=1)
        ttk.Label(left,textvariable=self.frame_status,wraplength=600).grid(row=0,column=0,sticky="w",pady=5)
        self.canvas=tk.Canvas(left,background="#dce2e5",highlightthickness=0,width=580,height=290)
        self.canvas.grid(row=1,column=0,sticky="nsew")
        self.canvas.bind("<Configure>",lambda e:self._draw_preview())
        actions=ttk.Frame(left,padding=(0,8))
        actions.grid(row=2,column=0,sticky="ew")
        self.capture_button=self._button(actions,"Chụp + đóng băng",self.capture,True,side="left",padx=(0,5))
        self.flip_button=self._button(actions,"Đảo gốc 180°",self.flip_origin,side="left",padx=5)
        self.diagnostic_button=self._button(actions,"Lưu scan lỗi",self.save_diagnostic,side="left",padx=5)
        ttk.Checkbutton(left,text="Hiện góc và trục bảng",variable=self.show_corners,command=self._update_preview).grid(row=3,column=0,sticky="w")
        right=ttk.LabelFrame(top,text="Pose tại lúc chụp · mặt bích / BASE",padding=10)
        right.grid(row=0,column=1,sticky="nsew")
        right.columnconfigure((0,1,2),weight=1)
        for i,label in enumerate(("X · mm","Y · mm","Z · mm","A · độ","B · độ","C · độ")):
            self._entry(right,label,self.pose_vars[i],(i//3)*2,i%3,12)
        paste=ttk.Frame(right)
        paste.grid(row=4,column=0,columnspan=3,sticky="ew",pady=4)
        ttk.Entry(paste,textvariable=self.paste_pose).pack(side="left",fill="x",expand=True)
        self._button(paste,"Dán 6 số",self.apply_pose,side="right",padx=(5,0))
        ttk.Label(right,text="Nhập X Y Z A B C · Euler ZYZ · dấu chấm thập phân",wraplength=420).grid(row=5,column=0,columnspan=3,sticky="w",pady=4)
        self.role_box=ttk.Combobox(right,textvariable=self.role,values=("Tính calibration","Kiểm chứng độc lập"),state="readonly")
        self.role_box.grid(row=6,column=0,columnspan=3,sticky="ew",pady=5)
        ttk.Checkbutton(right,text="Robot đứng yên, pose đúng ảnh đang xem",variable=self.stationary,command=self._refresh_buttons).grid(row=7,column=0,columnspan=3,sticky="w",pady=3)
        ttk.Checkbutton(right,text="Gốc / trục bảng đúng dấu vật lý trên bảng",variable=self.origin_ok,command=self._refresh_buttons).grid(row=8,column=0,columnspan=3,sticky="w",pady=3)
        savebox=ttk.Frame(right)
        savebox.grid(row=9,column=0,columnspan=3,sticky="ew",pady=6)
        self.save_button=self._button(savebox,"Xác nhận + lưu cặp ảnh / pose",self.save_pair,True,fill="x")
        ttk.Label(right,textvariable=self.pair_status,wraplength=420).grid(row=10,column=0,columnspan=3,sticky="w",pady=4)
        bottom=ttk.LabelFrame(pane,text="Mẫu đã thu · dữ liệu gốc vẫn giữ khi bỏ chọn",padding=8)
        pane.add(bottom,weight=2)
        bar=ttk.Frame(bottom)
        bar.pack(fill="x",pady=(0,5))
        ttk.Label(bar,textvariable=self.count_text).pack(side="left")
        self.toggle_button=self._button(bar,"Dùng / bỏ chọn mẫu",self.toggle_sample,side="right",padx=5)
        self._button(bar,"Sang tính kết quả",lambda:self.book.select(2),side="right",padx=5)
        columns=("id","role","use","corners","rms","time")
        self.tree=ttk.Treeview(bottom,columns=columns,show="headings",height=5,selectmode="browse")
        for c,title,width in zip(columns,("Mã mẫu","Vai trò","Dùng","Góc 3D","RMS bảng (mm)","Chụp lúc (UTC)"),(90,150,60,100,120,230)):
            self.tree.heading(c,text=title);self.tree.column(c,width=width,minwidth=50)
        scroll=ttk.Scrollbar(bottom,orient="vertical",command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right",fill="y");self.tree.pack(fill="both",expand=True)
        self.tree.bind("<Double-1>",lambda e:self.review_sample())
        self.tree.bind("<Return>",lambda e:self.review_sample())

    def _results_tab(self):
        tab=self.tabs[2]
        ttk.Label(tab,text="T_base_camera · Camera Space → BASE · tịnh tiến mm",font=("DejaVu Sans",11,"bold")).pack(anchor="w",pady=5)
        self.matrix=tk.Text(tab,height=6,width=80,font=("DejaVu Sans Mono",12),background="#ffffff",foreground="#202b35",wrap="none")
        self.matrix.pack(fill="x",pady=8)
        self.matrix.configure(state="disabled")
        actions=ttk.Frame(tab)
        actions.pack(fill="x",pady=6)
        self.solve_button=self._button(actions,"Tính calibration",self.calculate,True,side="left",padx=(0,6))
        self.export_button=self._button(actions,"Xuất JSON",self.export,side="left",padx=6)
        self._button(actions,"Sang kiểm chứng",lambda:self.book.select(3),side="left",padx=6)
        ttk.Label(tab,textvariable=self.result_text,wraplength=1050).pack(anchor="w",pady=8)
        self.error_tree=ttk.Treeview(tab,columns=("id","role","pos","rot","review"),show="headings",height=8)
        for c,title,width in zip(("id","role","pos","rot","review"),("Mã mẫu","Vai trò","Lệch vị trí (mm)","Lệch hướng (độ)","Đánh giá"),(90,150,160,160,240)):
            self.error_tree.heading(c,text=title);self.error_tree.column(c,width=width)
        self.error_tree.tag_configure("review",background="#fff0db",foreground="#55370e")
        self.error_tree.pack(fill="both",expand=True,pady=5)
        ttk.Label(tab,text="Mẫu có lỗi lớn không bị tự xóa. Kiểm tra gốc bảng, ZYZ và pose; bỏ chọn ở tab Thu mẫu rồi tính lại.",wraplength=1050).pack(anchor="w",pady=6)

    def _validation_tab(self):
        tab=self.tabs[3]
        ttk.Label(tab,text="Kiểm chứng ở các pose MỚI, không đưa vào tập tính",font=("DejaVu Sans",12,"bold")).pack(anchor="w",pady=8)
        ttk.Label(tab,textvariable=self.validation_text,wraplength=1050,font=("DejaVu Sans",11)).pack(anchor="w",pady=12)
        actions=ttk.Frame(tab)
        actions.pack(fill="x",pady=10)
        self._button(actions,"Thu mẫu kiểm chứng mới",self.start_validation,True,side="left",padx=(0,8))
        self._button(actions,"Tính lại và đánh giá",self.calculate,side="left",padx=8)
        self._button(actions,"Xuất JSON hiện tại",self.export,side="left",padx=8)
        text=("Kết quả lưu cả ma trận PhoXi → BASE và quan hệ bảng → mặt bích.\n\n"
              "Giữ nguyên camera so với BASE sau hiệu chuẩn. Đổi TCP gắp không làm đổi ma trận camera → BASE.\n\n"
              "Đạt ngưỡng kiểm chứng chỉ đánh giá calibration trên các mẫu đã thu; chưa chứng nhận TCP gắp, tiếp xúc hoặc quỹ đạo.\n\n"
              "Mọi kết quả DEMO được đánh dấu rõ và không được xem là calibration thực.\n\n"
              "Công cụ này không kết nối điều khiển chuyển động robot và không thay đổi cấu hình TCP.")
        ttk.Label(tab,text=text,wraplength=1000).pack(anchor="w",pady=16)

    def _run(self, label, work, done):
        if self.busy:
            return
        self.busy=True;self.status.set(label);self.progress.start(12);self._refresh_buttons()
        future=self.pool.submit(work)
        def poll():
            if not future.done():
                self.root.after(80,poll);return
            self.busy=False;self.progress.stop()
            try:
                done(future.result())
            except Exception as exc:
                self.status.set(str(exc))
                messagebox.showerror("Không hoàn thành thao tác",str(exc),parent=self.root)
            self._refresh_buttons()
        self.root.after(80,poll)

    def _refresh_buttons(self):
        if not hasattr(self,"save_button"):
            return
        for button in self.buttons:
            button.configure(state="disabled" if self.busy else "normal")
        if self.busy:
            return
        active=self.session is not None
        self.capture_button.configure(state="normal" if active else "disabled")
        self.flip_button.configure(state="normal" if self.frame is not None and not self.reviewing else "disabled")
        self.diagnostic_button.configure(state="normal" if self.frame is not None and active else "disabled")
        self.toggle_button.configure(state="normal" if active else "disabled")
        self.solve_button.configure(state="normal" if active else "disabled")
        self.export_button.configure(state="normal" if active and self.session.doc.get("current_result") else "disabled")
        try:
            parse_pose(" ".join(v.get() for v in self.pose_vars))
            valid_pose=True
        except ValueError:
            valid_pose=False
        can_save=active and self.detection is not None and not self.reviewing and self.stationary.get() and self.origin_ok.get() and valid_pose
        self.save_button.configure(state="normal" if can_save else "disabled")
        for widget in self.locked_widgets:
            widget.configure(state="disabled" if active else "normal")
        self.new_button.configure(state="disabled" if active else "normal")

    def _set_source(self):
        if self.source is not None:
            old=self.source
            self.pool.submit(old.close)
        self.config["camera"]["device_id"]=self.device.get().strip() or None
        self.source=DemoSource() if self.demo else PhoXiSource(self.config)
        if self.demo and self.session:
            self.source.index=1+max([int(s.get("frame_id",-1)) for s in self.session.doc["samples"]] or [-1])

    def new_session(self):
        if self.busy or self.session is not None:
            return
        try:
            spec=BoardSpec(int(self.cols.get()),int(self.rows.get()),float(self.square.get()))
            name=self.session_name.get().strip()
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}",name):
                raise ValueError("Tên phiên dùng chữ không dấu, số, dấu gạch dưới / gạch ngang.")
            limits=dict(self.options.get("solver",{}))
            limits.update(max_position_error_mm=float(self.pos_limit.get()),max_rotation_error_deg=float(self.rot_limit.get()))
            if not all(np.isfinite(limits[k]) and limits[k]>0 for k in ("max_position_error_mm","max_rotation_error_deg")):
                raise ValueError("Ngưỡng sai số phải là số hữu hạn > 0.")
            root=Path(self.output_dir.get()).expanduser()/name
            self.session=Session.create(root,spec,self.options.get("detection",{}),limits,self.demo)
            self._set_source();self._session_changed();self.book.select(1)
        except Exception as exc:
            messagebox.showerror("Không tạo được phiên",str(exc),parent=self.root)

    def prepare_new(self):
        if self.busy:return
        self.session=None;self.frame=self.detection=None;self.preview_image=None;self.reviewing=False
        self.session_name.set(self._new_name());self.stationary.set(False);self.origin_ok.set(False)
        self._session_changed();self._update_preview();self.book.select(0)

    def open_session(self):
        if self.busy:return
        path=filedialog.askopenfilename(title="Chọn session.json",filetypes=[("Phiên hiệu chuẩn","session.json")],parent=self.root)
        if path:self._load_session(path)

    def _load_session(self,path):
        try:
            session=Session.load(path)
            if bool(session.doc["demo"]) != self.demo:
                raise ValueError("Phiên DEMO cần chạy --demo; phiên camera thật phải chạy không có --demo.")
            self.session=session
            self.cols.set(str(session.spec.cols));self.rows.set(str(session.spec.rows));self.square.set(str(session.spec.square_mm))
            self.session_name.set(session.root.name);self.output_dir.set(str(session.root.parent))
            if session.doc.get("camera_identity"):
                self.device.set(session.doc["camera_identity"]["device_id"])
            self.pos_limit.set(str(session.doc["solver_settings"].get("max_position_error_mm",2)))
            self.rot_limit.set(str(session.doc["solver_settings"].get("max_rotation_error_deg",1)))
            self.frame=self.detection=None;self.reviewing=False;self.preview_image=None
            self._set_source();self._session_changed();self._show_result(session.current_result());self.book.select(1)
        except Exception as exc:
            messagebox.showerror("Không mở được phiên",str(exc),parent=self.root)

    def _session_changed(self):
        self.tree.delete(*self.tree.get_children());self.error_tree.delete(*self.error_tree.get_children())
        samples=self.session.doc["samples"] if self.session else []
        for s in samples:
            self.tree.insert("","end",iid=s["id"],values=(s["id"],"Tính" if s["role"]=="fit" else "Kiểm chứng",
                "Có" if s.get("enabled",True) else "Bỏ",f"{s['inlier_corners']}/{s['total_corners']}",
                f"{s['rms_mm']:.3f}",s.get("captured_at_utc","")))
        fit=sum(s["role"]=="fit" and s.get("enabled",True) for s in samples)
        val=sum(s["role"]=="validation" and s.get("enabled",True) for s in samples)
        self.count_text.set(f"{fit} mẫu tính · {val} mẫu kiểm chứng · nhấp đúp mã mẫu để xem lại")
        self.session_label.configure(text=str(self.session.root) if self.session else "Chưa mở phiên")
        self.status.set(("DEMO · " if self.demo else "")+(str(self.session.root) if self.session else "Tạo hoặc mở phiên để bắt đầu."))
        self._show_result(self.session.current_result() if self.session else None)
        self._refresh_buttons()

    def apply_pose(self):
        try:
            values=parse_pose(self.paste_pose.get())
            for var,value in zip(self.pose_vars,values):var.set(f"{value:.6f}")
            self.stationary.set(False);self._refresh_buttons()
        except ValueError as exc:messagebox.showerror("Pose không hợp lệ",str(exc),parent=self.root)

    def capture(self):
        if self.busy or self.session is None:return
        spec=self.session.spec;settings=self.session.doc["detection_settings"]
        self.frame=self.detection=None;self.reviewing=False;self.flipped.set(False)
        self.stationary.set(False);self.origin_ok.set(False)
        for var in self.pose_vars:var.set("")
        def work():
            frame=self.source.capture(spec)
            try:
                detection=estimate_board(frame,spec,settings=settings)
                if self.demo:
                    origin=np.asarray(frame.info["demo_origin_uv"])
                    if np.linalg.norm(detection.corners_uv[-1]-origin)<np.linalg.norm(detection.corners_uv[0]-origin):
                        detection=estimate_board(frame,spec,flipped=True,settings=settings,corners_uv=detection.corners_uv)
                return frame,detection,None
            except ValueError as exc:
                return frame,None,str(exc)
        def done(value):
            self.frame,self.detection,error=value
            self.flipped.set(bool(self.detection and self.detection.flipped))
            if self.demo:
                for var,v in zip(self.pose_vars,self.frame.info["demo_robot_pose_xyzabc"]):var.set(f"{v:.9f}")
            self._update_preview()
            if error:
                self.frame_status.set("Đã chụp · chưa xác định được pose bảng")
                self.pair_status.set(error);self.status.set("Có thể lưu scan lỗi để kiểm tra offline.")
            else:
                d=self.detection
                self.frame_status.set(f"Đóng băng frame {self.frame.frame_id} · 3D {d.inliers.sum()}/{len(d.valid)} góc · RMS {d.rms_mm:.3f} mm · cạnh ô {d.measured_square_mm:.2f} mm")
                self.pair_status.set("Nhập pose mặt bích; kiểm tra gốc màu vàng và hai trục X/Y.")
                self.status.set("Giữ robot ở đúng pose đến khi lưu mẫu xong.")
        self._run("Đang chụp và nhận diện bảng…",work,done)

    def flip_origin(self):
        if self.busy or self.frame is None or self.reviewing:return
        flipped=not self.flipped.get();self.origin_ok.set(False)
        frame,spec=self.frame,self.session.spec
        self._run("Đang đảo gốc bảng trên cùng khung ảnh…",
            lambda:estimate_board(frame,spec,flipped=flipped,settings=self.session.doc["detection_settings"]),
            self._flipped_done)

    def _flipped_done(self,detection):
        self.detection=detection;self.flipped.set(detection.flipped);self._update_preview()
        self.pair_status.set("Đã đảo gốc. Kiểm tra lại dấu vật lý và xác nhận gốc bảng.")

    def _update_preview(self):
        if self.frame is None:
            self.preview_image=None;self.canvas.delete("all");return
        image=overlay(self.frame,self.detection,self.session.spec) if self.show_corners.get() else self.frame.texture_bgr
        self.preview_image=Image.fromarray(image[:,:,::-1]);self._draw_preview()

    def _draw_preview(self):
        if self.preview_image is None:return
        width,height=max(20,self.canvas.winfo_width()),max(20,self.canvas.winfo_height())
        image=self.preview_image.copy();image.thumbnail((width,height),Image.Resampling.LANCZOS)
        self.tk_image=ImageTk.PhotoImage(image,master=self.root)
        self.canvas.delete("all");self.canvas.create_image(width/2,height/2,image=self.tk_image,anchor="center")

    def save_pair(self):
        if self.busy or self.detection is None or self.reviewing or self.session is None:return
        if not self.stationary.get() or not self.origin_ok.get():return
        try:pose=parse_pose(" ".join(v.get() for v in self.pose_vars))
        except ValueError as exc:messagebox.showerror("Pose không hợp lệ",str(exc),parent=self.root);return
        frame,detection,session=self.frame,self.detection,self.session
        role="fit" if self.role.get()=="Tính calibration" else "validation"
        def done(sample_id):
            self.reviewing=True;self.stationary.set(False);self.origin_ok.set(False)
            self.pair_status.set(f"Đã lưu {sample_id}. Chuyển robot sang pose mới rồi chụp ảnh mới.")
            self._session_changed();self.tree.selection_set(sample_id);self.tree.see(sample_id)
        self._run("Đang lưu ảnh + 3D + pose…",lambda:session.add(frame,detection,pose,role,True,True),done)

    def review_sample(self):
        if self.busy or self.session is None or not self.tree.selection():return
        sample_id=self.tree.selection()[0]
        def done(value):
            self.frame,self.detection,item=value;self.reviewing=True
            self.stationary.set(False);self.origin_ok.set(False)
            for var,v in zip(self.pose_vars,item["robot_pose_xyzabc"]):var.set(str(v))
            self._update_preview();self.frame_status.set("Xem lại "+sample_id+" · không phải khung mới")
            self.pair_status.set("Ảnh gốc và pose đã lưu. Bấm Chụp để thu mẫu mới.")
        self._run("Đang mở mẫu "+sample_id,lambda:self.session.read_sample(sample_id),done)

    def toggle_sample(self):
        if self.busy or self.session is None or not self.tree.selection():return
        sample_id=self.tree.selection()[0]
        self._run("Đang cập nhật danh sách mẫu…",lambda:self.session.toggle(sample_id),lambda _:self._session_changed())

    def calculate(self):
        if self.busy or self.session is None:return
        session=self.session
        def work():
            result=solve(session.doc["samples"],session.doc["solver_settings"])
            return session.store_result(result)
        def done(result):
            self._show_result(result);self.book.select(2);self.status.set("Đã lưu kết quả và bảng sai số vào thư mục phiên.")
        self._run("Đang tính ma trận PhoXi → BASE và kiểm chứng…",work,done)

    def _show_result(self,result):
        self.matrix.configure(state="normal");self.matrix.delete("1.0","end")
        self.error_tree.delete(*self.error_tree.get_children())
        if result is None:
            self.matrix.insert("end","Chưa có kết quả cho bộ mẫu hiện tại.")
            self.result_text.set("Chưa tính hoặc bộ mẫu đã thay đổi; cần tính lại.")
            self.validation_text.set("Chưa có kết quả kiểm chứng hiện tại.")
        else:
            t=np.asarray(result["T_base_camera"])
            self.matrix.insert("end","\n".join("  ".join(f"{v:13.6f}" for v in row) for row in t))
            f,v=result["fit"],result["validation"]
            self.result_text.set(f"{result['status']}\nTập tính {f['count']} mẫu · RMS vị trí {f['position_rms_mm']:.3f} mm · max {f['position_max_mm']:.3f} mm · RMS góc {f['rotation_rms_deg']:.3f}°.\n"+
                ("Cần xem: "+", ".join(result["issues"]) if result["issues"] else "Các mẫu nằm trong ngưỡng đã cấu hình.")+"\nKết quả không bao gồm TCP gắp / quỹ đạo robot.")
            if v["count"]:
                self.validation_text.set(f"{v['count']} mẫu độc lập · RMS vị trí {v['position_rms_mm']:.3f} mm · max {v['position_max_mm']:.3f} mm\nRMS góc {v['rotation_rms_deg']:.3f}° · max {v['rotation_max_deg']:.3f}°\n"+
                    ("DEMO — không dùng trên robot." if result.get("demo") else result["status"]))
            else:self.validation_text.set("Chưa có mẫu kiểm chứng độc lập. Thu ≥5 pose mới để đánh giá.")
            for row in result["residuals"]:
                self.error_tree.insert("","end",values=(row["id"],"Tính" if row["role"]=="fit" else "Kiểm chứng",
                    f"{row['position_error_mm']:.3f}",f"{row['rotation_error_deg']:.3f}","Cần xem lại" if row["review_required"] else "Trong ngưỡng"),tags=("review",) if row["review_required"] else ())
        self.matrix.configure(state="disabled")

    def start_validation(self):
        self.role.set("Kiểm chứng độc lập");self.book.select(1)

    def export(self):
        if self.busy or self.session is None:return
        result=self.session.current_result()
        if result is None:
            messagebox.showinfo("Chưa có kết quả","Tính lại calibration cho bộ mẫu hiện tại trước khi xuất.",parent=self.root);return
        name="DEMO_camera_to_base.json" if self.demo else "phoxi_camera_to_base.json"
        path=filedialog.asksaveasfilename(title="Xuất ma trận và báo cáo JSON",initialdir=self.session.root,initialfile=name,defaultextension=".json",filetypes=[("JSON","*.json")],parent=self.root)
        if path:self._run("Đang xuất JSON…",lambda:self.session.export(path),lambda _:self.status.set("Đã xuất "+path))

    def save_diagnostic(self):
        if self.busy or self.frame is None or self.session is None:return
        frame=self.frame;spec=self.session.spec;reason=self.pair_status.get()
        folder=self.session.root/"diagnostics"/datetime.now(timezone.utc).strftime("scan_%Y%m%dT%H%M%S_%fZ")
        def work():
            save_capture(frame,folder,stem="frame",save_3d=True)
            atomic_json(folder/"diagnostic.json",{"reason":reason,"board":spec.to_dict(),"demo":self.demo})
            return folder
        self._run("Đang lưu scan để kiểm tra offline…",work,lambda p:self.status.set("Đã lưu "+str(p)))

    def close(self):
        if self.source is not None:
            self.pool.submit(self.source.close)
        self.pool.shutdown(wait=False)
        self.root.destroy()
