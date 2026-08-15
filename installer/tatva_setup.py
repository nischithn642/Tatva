"""TATVA Setup — the Windows installer wizard.

This is a self-extracting installer in the same shape as Vivado's or any other
desktop installer: welcome, destination, options, a progress bar that means
something, and a finish page. It registers itself in Apps & features so it can be
uninstalled the normal way.

How the payload gets here
-------------------------
build_installer.py freezes this script with PyInstaller into a small stub, then
appends the distribution zip to the end of that exe followed by a fixed-size
footer holding the offset and length. At run time the installer opens its own
file, seeks to the offset, and reads the zip in place -- so a 240 MB payload is
never unpacked to a temp directory first. That is why the window appears
instantly rather than after a minute of invisible extraction.

Everything installs under the user's own profile, so there is no UAC prompt and
no administrator account required.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import shutil
import struct
import subprocess
import sys
import threading
import tkinter as tk
import winreg
import zipfile
from pathlib import Path
from tkinter import filedialog, ttk

APP_NAME = "TATVA"
APP_FULL_NAME = "TATVA Optimization Studio"
# These two mirror DISPLAY_VERSION and __version__ in src/tatva/__init__.py. The
# wizard is frozen without the tatva package -- it is a 15 MB Tkinter stub, and
# importing the compiler into it would drag numpy, onnx and TVM along -- so it cannot
# read them at runtime. build_installer.py compares these literals against the package
# and refuses to build on a mismatch, which is what keeps the copy from drifting.
APP_VERSION = "2.1"
APP_BUILD = "2.1.0"
PUBLISHER = "TATVA"
EXE_NAME = "TATVA.exe"
REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\TATVA"

FOOTER_MAGIC = b"TATVAPKG1"
FOOTER_FORMAT = "<QQ"  # payload offset, payload length
FOOTER_SIZE = len(FOOTER_MAGIC) + struct.calcsize(FOOTER_FORMAT)

# Palette lifted from the app itself so the installer does not look like a
# different product than the thing it installs.
GOLD = "#C68A26"
GOLD_DIM = "#A9761F"
INK = "#16161A"
INK_2 = "#4A4A55"
INK_3 = "#7C7C88"
BG = "#F4F4F1"
PANEL = "#FFFFFF"
LINE = "#DCDCD6"
OK = "#1D7A45"
ERR = "#B3261E"


# --------------------------------------------------------------------------- payload


class PayloadSlice:
    """A read-only, seekable window onto part of a larger file.

    zipfile locates the central directory by seeking from the end, so it needs a
    file object whose end is the end of the zip -- not the end of the exe. This
    presents exactly the byte range the payload occupies.
    """

    def __init__(self, handle, offset: int, length: int) -> None:
        self._handle = handle
        self._offset = offset
        self._length = length
        self._pos = 0

    def read(self, size: int = -1) -> bytes:
        remaining = self._length - self._pos
        if remaining <= 0:
            return b""
        if size is None or size < 0 or size > remaining:
            size = remaining
        self._handle.seek(self._offset + self._pos)
        data = self._handle.read(size)
        self._pos += len(data)
        return data

    def seek(self, pos: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self._pos = pos
        elif whence == os.SEEK_CUR:
            self._pos += pos
        else:
            self._pos = self._length + pos
        self._pos = max(0, min(self._pos, self._length))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._handle.close()


def open_payload():
    """Return a file object holding the distribution zip, or raise."""
    # Frozen: the payload is appended to this exe.
    exe = Path(sys.executable)
    if getattr(sys, "frozen", False):
        handle = exe.open("rb")
        handle.seek(0, os.SEEK_END)
        total = handle.tell()
        if total > FOOTER_SIZE:
            handle.seek(total - FOOTER_SIZE)
            footer = handle.read(FOOTER_SIZE)
            if footer.startswith(FOOTER_MAGIC):
                offset, length = struct.unpack(FOOTER_FORMAT, footer[len(FOOTER_MAGIC) :])
                return PayloadSlice(handle, offset, length)
        handle.close()
        raise RuntimeError(
            "This installer is missing its payload. Re-download the setup file; a partial download is the usual cause."
        )

    # Running from source: accept a zip on the command line, or find the built one.
    for arg in sys.argv[1:]:
        if arg.lower().endswith(".zip") and Path(arg).exists():
            return Path(arg).open("rb")
    here = Path(__file__).resolve().parent.parent / "dist"
    candidates = sorted(here.glob("TATVA-*-windows.zip"))
    if candidates:
        return candidates[-1].open("rb")
    raise RuntimeError("No payload zip found. Pass one as an argument when running from source.")


# --------------------------------------------------------------------------- install


def default_install_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "Programs" / APP_NAME


def make_shortcut(link_path: Path, target: Path, working_dir: Path, description: str) -> None:
    """Create a .lnk. PowerShell's WScript.Shell is the only stdlib-free way to do this."""
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{link}');"
        "$s.TargetPath = '{target}';"
        "$s.WorkingDirectory = '{wd}';"
        "$s.Description = '{desc}';"
        "$s.IconLocation = '{target},0';"
        "$s.Save()"
    ).format(
        link=str(link_path).replace("'", "''"),
        target=str(target).replace("'", "''"),
        wd=str(working_dir).replace("'", "''"),
        desc=description.replace("'", "''"),
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        creationflags=0x08000000,  # CREATE_NO_WINDOW
        timeout=60,
    )


UNINSTALL_SCRIPT = r"""# TATVA uninstaller.
# Re-launches itself from the temp folder, because a script cannot delete the
# directory it is running from.
param([switch]$Relaunched)

$InstallDir = '{install_dir}'
$StartMenu  = '{start_menu}'
$Desktop    = '{desktop}'

if (-not $Relaunched) {{
    $copy = Join-Path $env:TEMP 'tatva-uninstall.ps1'
    Copy-Item $PSCommandPath $copy -Force
    Start-Process powershell -ArgumentList @(
        '-NoProfile','-ExecutionPolicy','Bypass','-File',$copy,'-Relaunched'
    ) -WindowStyle Hidden
    return
}}

# Give the launching shell a moment to release the original file.
Start-Sleep -Milliseconds 800

foreach ($lnk in @($StartMenu, $Desktop)) {{
    if ($lnk -and (Test-Path $lnk)) {{ Remove-Item $lnk -Force -ErrorAction SilentlyContinue }}
}}

$menuFolder = Split-Path $StartMenu -Parent
if ($menuFolder -and (Test-Path $menuFolder)) {{
    if (-not (Get-ChildItem $menuFolder -Force)) {{
        Remove-Item $menuFolder -Recurse -Force -ErrorAction SilentlyContinue
    }}
}}

if (Test-Path $InstallDir) {{
    Remove-Item $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
}}

Remove-Item 'HKCU:\{reg_key}' -Recurse -Force -ErrorAction SilentlyContinue
"""


def write_uninstaller(install_dir: Path, start_menu: Path | None, desktop: Path | None) -> Path:
    path = install_dir / "uninstall.ps1"
    path.write_text(
        UNINSTALL_SCRIPT.format(
            install_dir=str(install_dir).replace("'", "''"),
            start_menu=str(start_menu).replace("'", "''") if start_menu else "",
            desktop=str(desktop).replace("'", "''") if desktop else "",
            reg_key=REG_KEY,
        ),
        encoding="utf-8",
    )
    return path


def register_in_apps_and_features(install_dir: Path, size_kb: int) -> None:
    uninstall_cmd = f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{install_dir / "uninstall.ps1"}"'
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, REG_KEY) as key:
        values = {
            "DisplayName": f"{APP_FULL_NAME} ({APP_VERSION})",
            "DisplayVersion": APP_BUILD,
            "Publisher": PUBLISHER,
            "InstallLocation": str(install_dir),
            "DisplayIcon": str(install_dir / EXE_NAME),
            "UninstallString": uninstall_cmd,
            "QuietUninstallString": uninstall_cmd,
        }
        for name, value in values.items():
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        winreg.SetValueEx(key, "EstimatedSize", 0, winreg.REG_DWORD, size_kb)
        winreg.SetValueEx(key, "NoModify", 0, winreg.REG_DWORD, 1)
        winreg.SetValueEx(key, "NoRepair", 0, winreg.REG_DWORD, 1)


def unregister() -> None:
    with contextlib.suppress(OSError):
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, REG_KEY)


# --------------------------------------------------------------------------- wizard


class Setup(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_FULL_NAME} Setup")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.install_dir = tk.StringVar(value=str(default_install_dir()))
        self.want_start_menu = tk.BooleanVar(value=True)
        self.want_desktop = tk.BooleanVar(value=True)
        self.want_launch = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="")
        self.detail = tk.StringVar(value="")

        self.page = 0
        self.installing = False
        self.finished = False
        self.failure: str | None = None
        self.installed_bytes = 0

        self._center(760, 500)
        self._style()
        self._build()
        self.show_page(0)

        # Enter advances, Escape backs out -- the keyboard behaviour every other
        # installer has, and the reason this window can be driven without a mouse.
        self.bind("<Return>", self._on_return)
        self.bind("<Escape>", lambda _event: self.on_close())

    # ---------------------------------------------------------------- chrome

    def _center(self, w: int, h: int) -> None:
        with contextlib.suppress(Exception):  # crisp text on a scaled display
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 3
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=PANEL)
        s.configure("Side.TFrame", background=INK)
        s.configure("Foot.TFrame", background=BG)
        s.configure("TLabel", background=PANEL, foreground=INK, font=("Segoe UI", 10))
        s.configure("H1.TLabel", font=("Segoe UI Semibold", 17), foreground=INK)
        s.configure("Sub.TLabel", font=("Segoe UI", 10), foreground=INK_2)
        s.configure("Tiny.TLabel", font=("Segoe UI", 9), foreground=INK_3)
        s.configure("Kicker.TLabel", font=("Consolas", 9), foreground=GOLD_DIM)
        s.configure("TCheckbutton", background=PANEL, foreground=INK, font=("Segoe UI", 10))
        s.map("TCheckbutton", background=[("active", PANEL)])
        s.configure("TEntry", fieldbackground="#FFFFFF", bordercolor=LINE)
        s.configure(
            "Gold.Horizontal.TProgressbar",
            troughcolor="#E8E8E3",
            background=GOLD,
            bordercolor="#E8E8E3",
            lightcolor=GOLD,
            darkcolor=GOLD,
            thickness=8,
        )

    def _build(self) -> None:
        # Left band: the product identity, so the window is recognisable at a glance.
        side = tk.Frame(self, bg=INK, width=210)
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        tk.Frame(side, bg=GOLD, height=4, width=54).place(x=26, y=44)
        tk.Label(side, text="TATVA", bg=INK, fg="#FFFFFF", font=("Segoe UI Semibold", 24)).place(x=24, y=60)
        tk.Label(
            side,
            text="Optimization Studio",
            bg=INK,
            fg="#B9B9C4",
            font=("Segoe UI", 10),
        ).place(x=26, y=100)
        tk.Label(side, text=APP_VERSION.upper(), bg=INK, fg=GOLD, font=("Consolas", 9)).place(x=26, y=126)
        tk.Label(
            side,
            text="Bare-metal RISC-V\ncompilation and\nbenchmarking.\n\nRuns entirely on\nthis machine.",
            bg=INK,
            fg="#84848F",
            font=("Segoe UI", 9),
            justify="left",
        ).place(x=26, y=330)

        self.body = tk.Frame(self, bg=PANEL)
        self.body.pack(side="top", fill="both", expand=True)

        foot = tk.Frame(self, bg=BG, height=62)
        foot.pack(side="bottom", fill="x")
        foot.pack_propagate(False)
        tk.Frame(foot, bg=LINE, height=1).pack(side="top", fill="x")

        self.btn_cancel = tk.Button(foot, text="Cancel", command=self.on_close, **self._btn(False))
        self.btn_cancel.pack(side="right", padx=(0, 22), pady=13)
        self.btn_next = tk.Button(foot, text="Next", command=self.on_next, **self._btn(True))
        self.btn_next.pack(side="right", padx=(0, 10), pady=13)
        self.btn_back = tk.Button(foot, text="Back", command=self.on_back, **self._btn(False))
        self.btn_back.pack(side="right", padx=(0, 10), pady=13)

        self.pages = [self._page_welcome(), self._page_location(), self._page_progress(), self._page_done()]

    def _btn(self, primary: bool) -> dict:
        return dict(
            font=("Segoe UI", 10),
            width=12,
            relief="flat",
            cursor="hand2",
            bg=GOLD if primary else "#FFFFFF",
            fg="#FFFFFF" if primary else INK,
            activebackground=GOLD_DIM if primary else "#F0F0EC",
            activeforeground="#FFFFFF" if primary else INK,
            highlightthickness=0 if primary else 1,
            highlightbackground=LINE,
            bd=0,
            pady=6,
        )

    def _frame(self) -> tk.Frame:
        f = tk.Frame(self.body, bg=PANEL)
        return f

    # ---------------------------------------------------------------- pages

    def _page_welcome(self) -> tk.Frame:
        f = self._frame()
        tk.Label(f, text="SETUP", bg=PANEL, fg=GOLD_DIM, font=("Consolas", 9)).pack(anchor="w", pady=(44, 4))
        tk.Label(
            f,
            text=f"Install {APP_FULL_NAME}",
            bg=PANEL,
            fg=INK,
            font=("Segoe UI Semibold", 18),
        ).pack(anchor="w")
        tk.Label(
            f,
            text=(
                "TATVA compiles ONNX models into standalone C programs for bare-metal\n"
                "RISC-V, then measures them under emulation."
            ),
            bg=PANEL,
            fg=INK_2,
            font=("Segoe UI", 10),
            justify="left",
        ).pack(anchor="w", pady=(10, 0))

        box = tk.Frame(f, bg="#FBF7EF", highlightbackground="#E8D3A8", highlightthickness=1)
        box.pack(anchor="w", fill="x", pady=(22, 0))
        tk.Label(
            box,
            text="Everything is included",
            bg="#FBF7EF",
            fg=INK,
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w", padx=14, pady=(11, 2))
        tk.Label(
            box,
            text=(
                "The RISC-V cross-compiler and the QEMU emulator are bundled. Nothing is\n"
                "downloaded during or after installation, and the app never sends your\n"
                "models anywhere. It works on a machine with no internet connection."
            ),
            bg="#FBF7EF",
            fg=INK_2,
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w", padx=14, pady=(0, 12))

        tk.Label(
            f,
            text=(
                "Installs for the current user only, so no administrator password is needed.\n"
                "Requires about 750 MB of free disk space."
            ),
            bg=PANEL,
            fg=INK_3,
            font=("Segoe UI", 9),
            justify="left",
        ).pack(anchor="w", pady=(20, 0))
        return f

    def _page_location(self) -> tk.Frame:
        f = self._frame()
        tk.Label(f, text="DESTINATION", bg=PANEL, fg=GOLD_DIM, font=("Consolas", 9)).pack(anchor="w", pady=(44, 4))
        tk.Label(f, text="Where to install", bg=PANEL, fg=INK, font=("Segoe UI Semibold", 18)).pack(anchor="w")
        tk.Label(
            f,
            text="Setup will create this folder and place the application inside it.",
            bg=PANEL,
            fg=INK_2,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 0))

        row = tk.Frame(f, bg=PANEL)
        row.pack(anchor="w", fill="x", pady=(18, 0))
        entry = tk.Entry(
            row,
            textvariable=self.install_dir,
            font=("Segoe UI", 10),
            relief="flat",
            highlightthickness=1,
            highlightbackground=LINE,
            highlightcolor=GOLD,
            bg="#FFFFFF",
        )
        entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 8))
        tk.Button(row, text="Browse…", command=self.on_browse, **self._btn(False)).pack(side="left")

        self.space_label = tk.Label(f, text="", bg=PANEL, fg=INK_3, font=("Segoe UI", 9), justify="left")
        self.space_label.pack(anchor="w", pady=(9, 0))

        tk.Frame(f, bg=LINE, height=1).pack(fill="x", pady=(26, 20))
        tk.Label(f, text="Shortcuts", bg=PANEL, fg=INK, font=("Segoe UI Semibold", 11)).pack(anchor="w")
        ttk.Checkbutton(f, text="Add to the Start menu", variable=self.want_start_menu).pack(anchor="w", pady=(9, 0))
        ttk.Checkbutton(f, text="Create a desktop shortcut", variable=self.want_desktop).pack(anchor="w", pady=(3, 0))
        return f

    def _page_progress(self) -> tk.Frame:
        f = self._frame()
        tk.Label(f, text="INSTALLING", bg=PANEL, fg=GOLD_DIM, font=("Consolas", 9)).pack(anchor="w", pady=(44, 4))
        tk.Label(f, text="Setting up TATVA", bg=PANEL, fg=INK, font=("Segoe UI Semibold", 18)).pack(anchor="w")
        tk.Label(
            f,
            text="Unpacking the application and the bundled RISC-V toolchain.",
            bg=PANEL,
            fg=INK_2,
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(8, 0))

        self.bar = ttk.Progressbar(
            f, style="Gold.Horizontal.TProgressbar", mode="determinate", maximum=1000, length=470
        )
        self.bar.pack(anchor="w", fill="x", pady=(30, 10))
        tk.Label(f, textvariable=self.status, bg=PANEL, fg=INK, font=("Segoe UI", 10)).pack(anchor="w")
        tk.Label(f, textvariable=self.detail, bg=PANEL, fg=INK_3, font=("Consolas", 9), anchor="w").pack(
            anchor="w", fill="x", pady=(4, 0)
        )
        return f

    def _page_done(self) -> tk.Frame:
        f = self._frame()
        self.done_kicker = tk.Label(f, text="FINISHED", bg=PANEL, fg=GOLD_DIM, font=("Consolas", 9))
        self.done_kicker.pack(anchor="w", pady=(44, 4))
        self.done_title = tk.Label(f, text="", bg=PANEL, fg=INK, font=("Segoe UI Semibold", 18))
        self.done_title.pack(anchor="w")
        self.done_body = tk.Label(f, text="", bg=PANEL, fg=INK_2, font=("Segoe UI", 10), justify="left")
        self.done_body.pack(anchor="w", pady=(10, 0))
        self.done_extra = tk.Frame(f, bg=PANEL)
        self.done_extra.pack(anchor="w", fill="x", pady=(20, 0))
        return f

    def show_page(self, index: int) -> None:
        for p in self.pages:
            p.pack_forget()
        self.page = index
        self.pages[index].pack(fill="both", expand=True, padx=40)
        if index == 1:
            self.update_space()
        self.btn_back.configure(state="normal" if index == 1 else "disabled")
        if index == 0:
            self.btn_next.configure(text="Next", state="normal")
        elif index == 1:
            self.btn_next.configure(text="Install", state="normal")
        elif index == 2:
            self.btn_next.configure(text="Install", state="disabled")
            self.btn_back.configure(state="disabled")
        else:
            self.btn_next.configure(text="Finish", state="normal")
            self.btn_cancel.configure(state="disabled")

    # ---------------------------------------------------------------- actions

    def on_browse(self) -> None:
        chosen = filedialog.askdirectory(title="Choose an installation folder", mustexist=False)
        if chosen:
            path = Path(chosen)
            if path.name.lower() != APP_NAME.lower():
                path = path / APP_NAME
            self.install_dir.set(str(path))
            self.update_space()

    def update_space(self) -> None:
        try:
            target = Path(self.install_dir.get())
            probe = target
            while not probe.exists() and probe.parent != probe:
                probe = probe.parent
            free = shutil.disk_usage(probe).free / (1024**3)
            note = f"About 750 MB required.  {free:.1f} GB free on this drive."
            if free < 1.0:
                note += "   — that is not enough room."
            self.space_label.configure(text=note, fg=ERR if free < 1.0 else INK_3)
        except Exception:
            self.space_label.configure(text="About 750 MB required.", fg=INK_3)

    def on_back(self) -> None:
        if self.page == 1:
            self.show_page(0)

    def on_next(self) -> None:
        if self.page == 0:
            self.show_page(1)
        elif self.page == 1:
            self.show_page(2)
            self.start_install()
        elif self.page == 3:
            if self.want_launch.get() and not self.failure:
                exe = Path(self.install_dir.get()) / EXE_NAME
                if exe.exists():
                    os.startfile(str(exe))
            self.destroy()

    def _on_return(self, _event=None) -> None:
        if str(self.btn_next["state"]) != "disabled":
            self.on_next()

    def on_close(self) -> None:
        if self.installing:
            return
        self.destroy()

    # ---------------------------------------------------------------- worker

    def start_install(self) -> None:
        self.installing = True
        threading.Thread(target=self._install_worker, daemon=True).start()

    def _progress(self, fraction: float, status: str, detail: str = "") -> None:
        def apply() -> None:
            self.bar["value"] = max(0, min(1000, int(fraction * 1000)))
            self.status.set(status)
            if detail:
                self.detail.set(detail if len(detail) <= 68 else "…" + detail[-67:])

        self.after(0, apply)

    def _install_worker(self) -> None:
        try:
            target = Path(self.install_dir.get())
            self._progress(0.0, "Preparing…", str(target))

            if target.exists() and any(target.iterdir()):
                # A previous install is in the way. Replace it rather than merging
                # two builds into one folder, which is how stale files survive.
                self._progress(0.02, "Removing the previous installation…")
                for child in target.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        with contextlib.suppress(OSError):
                            child.unlink()
            target.mkdir(parents=True, exist_ok=True)

            payload = open_payload()
            with zipfile.ZipFile(payload) as archive:
                members = [m for m in archive.infolist() if not m.is_dir()]
                total = sum(m.file_size for m in members) or 1
                written = 0

                for member in members:
                    # The zip carries a single TATVA/ prefix; drop it so the files
                    # land directly in the folder the user chose.
                    parts = Path(member.filename).parts
                    relative = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    with archive.open(member) as source, destination.open("wb") as sink:
                        shutil.copyfileobj(source, sink, 1024 * 256)

                    written += member.file_size
                    self._progress(
                        0.03 + 0.92 * (written / total),
                        f"Installing files…  {written / (1024**2):,.0f} MB of {total / (1024**2):,.0f} MB",
                        str(relative),
                    )
                self.installed_bytes = total
            payload.close()

            exe = target / EXE_NAME
            if not exe.exists():
                raise RuntimeError(f"{EXE_NAME} is missing from the installed files.")

            start_menu_lnk = None
            desktop_lnk = None

            if self.want_start_menu.get():
                self._progress(0.96, "Creating shortcuts…", "Start menu")
                folder = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / APP_NAME
                folder.mkdir(parents=True, exist_ok=True)
                start_menu_lnk = folder / f"{APP_FULL_NAME}.lnk"
                make_shortcut(start_menu_lnk, exe, target, APP_FULL_NAME)

            if self.want_desktop.get():
                self._progress(0.97, "Creating shortcuts…", "Desktop")
                desktop = Path(os.environ.get("USERPROFILE", Path.home())) / "Desktop"
                if desktop.is_dir():
                    desktop_lnk = desktop / f"{APP_NAME}.lnk"
                    make_shortcut(desktop_lnk, exe, target, APP_FULL_NAME)

            self._progress(0.99, "Registering…", "Apps & features")
            write_uninstaller(target, start_menu_lnk, desktop_lnk)
            register_in_apps_and_features(target, int(self.installed_bytes / 1024))

            self._progress(1.0, "Done.", "")
            self.after(250, self._finish_ok)

        except Exception as exc:  # surfaced on the finish page, never as a traceback
            self.failure = str(exc)
            self.after(0, self._finish_failed)

    def _finish_ok(self) -> None:
        self.installing = True  # keep Cancel inert; Finish closes the window
        self.done_kicker.configure(text="FINISHED", fg=OK)
        self.done_title.configure(text=f"{APP_NAME} is installed")
        self.done_body.configure(
            text=(
                f"Installed to  {self.install_dir.get()}\n\n"
                "Open the app and start at stage 01. Four sample models are included,\n"
                "so you can run the full five-stage pipeline straight away.\n\n"
                "TATVA-Stage-Guide.pdf in the installation folder explains what each\n"
                "stage does."
            )
        )
        for child in self.done_extra.winfo_children():
            child.destroy()
        ttk.Checkbutton(self.done_extra, text=f"Open {APP_NAME} now", variable=self.want_launch).pack(anchor="w")
        self.installing = False
        self.show_page(3)

    def _finish_failed(self) -> None:
        self.installing = False
        self.done_kicker.configure(text="STOPPED", fg=ERR)
        self.done_title.configure(text="Setup could not finish")
        self.done_body.configure(
            text=(
                f"{self.failure}\n\n"
                "Nothing was left running. You can close this window, resolve the problem,\n"
                "and run setup again."
            )
        )
        for child in self.done_extra.winfo_children():
            child.destroy()
        self.want_launch.set(False)
        self.show_page(3)


def main() -> int:
    Setup().mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
