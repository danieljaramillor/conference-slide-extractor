"""Small desktop controller for the local review app on Windows."""
import ctypes
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
import webbrowser


def launch(data_directory):
    # Only hide a console created for this double-click (one-file parent and child).
    kernel = ctypes.windll.kernel32
    members = (ctypes.c_ulong * 8)()
    count = kernel.GetConsoleProcessList(members, len(members))
    if count and set(members[:count]) <= {os.getpid(), os.getppid()}:
        kernel.GetConsoleWindow.restype = ctypes.c_void_p
        window = kernel.GetConsoleWindow()
        if window:
            ctypes.windll.user32.ShowWindow(ctypes.c_void_p(window), 0)

    root = tk.Tk()
    root.title('Conference Slide Extractor')
    root.resizable(False, False)
    panel = ttk.Frame(root, padding=24)
    panel.pack(fill='both', expand=True)
    ttk.Label(panel, text='Conference Slide Extractor', font=('Segoe UI', 16, 'bold')).pack(anchor='w')
    status = ttk.Label(panel, text='Starting the local workspace…', padding=(0, 12))
    status.pack(anchor='w')
    root.update()
    try:
        from batch_app import build_app
        server = build_app(data_directory)
    except Exception as exc:
        messagebox.showerror('Could not start the extractor', str(exc), parent=root)
        root.destroy()
        return 1
    url = f'http://127.0.0.1:{server.server_port}/'
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def open_app():
        if not webbrowser.open(url):
            status.configure(text='Copy the address below into your browser.')

    def quit_app():
        if server.batches.active and not messagebox.askyesno(
            'Stop extraction?', 'A batch is running. Quit and interrupt this batch?', default='no', parent=root
        ):
            return
        root.destroy()

    status.configure(text='Keep this window open while you work in your browser.')
    address = ttk.Entry(panel, width=46)
    address.insert(0, url)
    address.configure(state='readonly')
    address.pack(fill='x', pady=(0, 16))
    actions = ttk.Frame(panel)
    actions.pack(fill='x')
    ttk.Button(actions, text='Open extractor', command=open_app).pack(side='left')
    ttk.Button(actions, text='Quit', command=quit_app).pack(side='right')
    root.protocol('WM_DELETE_WINDOW', quit_app)
    root.after(150, open_app)
    try:
        root.mainloop()
    finally:
        server.shutdown()
        server.batches.close()
        server.server_close()
        thread.join()
