import argparse
import copy
import os
import sys
from enum import Enum

import cmasher as cmr
import datashader as ds
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import numpy as np
import polars as pl
from datashader.mpl_ext import dsshow
from matplotlib import gridspec
from matplotlib.backend_bases import _Mode
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.axisartist.axislines import AxesZero
from mpl_toolkits.axisartist.parasite_axes import HostAxes

# from mpl_toolkits.axisartist.parasite_axes import HostAxes
from param_manager.param_manager import ParamManager
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QObject, Qt, pyqtSlot
from PyQt5.QtGui import QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

plt.rcParams["axes.facecolor"] = (35 / 235, 35 / 235, 35 / 235)
plt.rcParams["figure.facecolor"] = (35 / 235, 35 / 235, 35 / 235)
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["axes.titlecolor"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["xtick.labelcolor"] = "white"
plt.rcParams["ytick.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"


class AppMode(Enum):
    GATE_EDIT = 0
    DELETE = 1
    GATE_SELECT = 2


class Gate:
    def __init__(self, z_lbl=0, a_lbl=0, low=0, high=1, left=0, right=1):
        self.z_lbl = z_lbl
        self.a_lbl = a_lbl

        self.low = low
        self.high = high
        self.left = left
        self.right = right

        self.locked = True
        self.interactor = None

    def set_left(self, value):
        if value > self.right:
            self.right += value - self.left
        self.left = value

    def set_right(self, value):
        if value < self.left:
            self.left += value - self.right
        self.right = value

    def set_bottom(self, value):
        self.low = value
        if self.low > self.high:
            self.low, self.high = self.high, self.low

    def set_top(self, value):
        self.high = value
        if self.low > self.high:
            self.low, self.high = self.high, self.low

    def x_center(self) -> float:
        return (self.left + self.right) / 2


class LinGateInteractor(QObject):
    def __init__(self, ax, gate, app_data):
        super().__init__()
        if ax is None:
            return
        self.gate = gate
        gate.interactor = self
        self.app_data = app_data
        self.ax = ax

        if ax is not None:
            self.canvas = ax.figure.canvas

        color = (0.0, 0, 0.3, 0.3)
        edge_color = "blue"
        if gate == app_data.active_gate:
            color = (0.3, 0, 0.0, 0.3)
            edge_color = "red"

        self.rect = Rectangle(
            [gate.left, gate.low],
            gate.right - gate.left,
            gate.high - gate.low,
            facecolor=color,
            edgecolor=edge_color,
            lw=0.5,
            animated=True,
        )

        self.ax.add_patch(self.rect)

    def update_visuals(self):
        self.rect.set_bounds(
            self.gate.left,
            self.gate.low,
            self.gate.right - self.gate.left,
            self.gate.high - self.gate.low,
        )
        color = (0.0, 0, 0.3, 0.3)
        edge_color = "blue"
        if self.gate == self.app_data.active_gate:
            color = (0.3, 0, 0.0, 0.3)
            edge_color = "red"
        self.rect.set_facecolor(color)
        self.rect.set_edgecolor(edge_color)

    def draw_callback(self, event):
        self.ax.add_patch(self.rect)
        self.ax.draw_artist(self.rect)
        return

    def motion_notify_callback(self, event):
        return

    def clean(self):
        self.rect.remove()
        pass


class InteractorManager(QObject):
    def __init__(self, fig, ax, interactors):
        super().__init__()

        self.interactors = interactors
        self.ax = ax
        self.fig = fig
        self.canvas = self.fig.canvas
        self.canvas.mpl_connect("draw_event", self.draw_callback)
        self.canvas.mpl_connect("motion_notify_event", self.motion_notify_callback)
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for interactor in self.interactors:
            if interactor is not None:
                interactor.draw_callback(event)

    def add_managed_artist(self, artist):
        self.interactors.append(artist)

    def motion_notify_callback(self, event):
        if event.inaxes is None:
            return

        else:
            self.canvas.restore_region(self.background)
            for interactor in self.interactors:
                interactor.motion_notify_callback(event)
                interactor.draw_callback(event)
            self.canvas.blit(self.ax.bbox)

    def clean(self):
        self.canvas.restore_region(self.background)
        for interactor in self.interactors:
            interactor.clean()
        self.canvas.blit(self.ax.bbox)


class AppData:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("directory")
        parser.add_argument("-i", "--input_file", default="ridges.parquet")
        parser.add_argument("-c", "--curve_file", default="curves.parquet")
        parser.add_argument("-l", "--limit", default=None, type=int)
        parser.add_argument("-b", "--bins", default=10_000, type=int)

        args = parser.parse_args()
        self.directory = args.directory
        self.raw_file = "linearized.parquet"
        self.mode_selector = None
        self.out_file = "lims.dat"
        self.bins = args.bins
        self.params = ParamManager(self.directory)

        self.mode = AppMode.GATE_EDIT

        self.lin_ax = None
        self.lin_fig = None
        self.lin_canvas = None

        raw_df_file = os.path.join(self.directory, self.raw_file)

        lf = pl.scan_parquet(raw_df_file).select(
            [self.params.x_col, self.params.y_col, "z_lin"]
        )
        if args.limit is not None:
            lf = lf.limit(args.limit)

        self.raw_df = lf.collect()
        self.raw_df_pd = self.raw_df.to_pandas
        self.active_gate = None
        self.gates = []
        self.lin_gate_interactors = []
        self.lin_interactor_manager = None

        self.vertical_mode = False

        self.lin_nav_bar = None

        selfpid_ax = None

        self.side_panel = None

        return

    def update_mode_selector(self):
        self.mode_selector.update_radio_btn()

    def update_pid_axes(self):
        ax_locations = [g.x_center() for g in self.gates]
        if self.pid_ax is not None:
            pid_ax_labels = [f"{g.z_lbl} | {g.a_lbl}" for g in self.gates]
            self.pid_ax.set_xticks(ax_locations)
            self.pid_ax.set_xticklabels(pid_ax_labels)
            plt.setp(self.pid_ax.axis["pid"].major_ticklabels, rotation=-90, ha="left")
        return

    def nearest_non_active_gate(self, x) -> Gate:
        nearest_gate = None
        nearest_dx = None
        for gate in self.saved_gates():
            dx = np.abs(gate.x_center() - x)
            if nearest_gate is None:
                nearest_gate = gate
                nearest_dx = dx
                continue
            elif dx < nearest_dx:
                nearest_gate = gate
                nearest_dx = dx
        return nearest_gate

    def saved_gates(self):
        return [g for g in self.gates if g != self.active_gate]

    def update_interactors(self):
        if self.lin_ax is not None:
            if self.lin_interactor_manager is not None:
                self.lin_interactor_manager.clean()
            self.lin_interactor_manager = InteractorManager(
                self.lin_fig,
                self.lin_ax,
                [LinGateInteractor(self.lin_ax, g, self) for g in self.gates],
            )

        self.update_canvases()
        return

    def update_canvases(self):
        self.lin_canvas.draw()

    def set_mode_to_edit(self):
        self.mode = AppMode.GATE_EDIT

    # def set_mode_to_delete(self):
    # self.mode = AppMode.DELETE

    def set_mode_to_gate_select(self):
        self.mode = AppMode.GATE_SELECT


class RawPanel(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QVBoxLayout(self)

        main_plot = RawCanvas(self.app_data, self, width=8, height=6)

        self.layout.addWidget(main_plot)
        self.app_data.nav_bar = NavigationToolbar(main_plot.canvas, self)
        self.layout.addWidget(self.app_data.nav_bar)

        self.setLayout(self.layout)


class LinPanel(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QVBoxLayout(self)

        main_plot = LinCanvas(self.app_data, self, width=8, height=6)

        self.layout.addWidget(main_plot)
        self.app_data.lin_nav_bar = NavigationToolbar(main_plot.canvas, self)
        self.layout.addWidget(self.app_data.lin_nav_bar)

        self.setLayout(self.layout)


class RawCanvas(FigureCanvas):
    def __init__(self, app_data, parent=None, width=8, height=6, dpi=100, arg=0):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        self.arg = arg

        self.app_data = app_data
        FigureCanvas.__init__(self, self.fig)
        self.canvas = self.fig.canvas
        self.setParent(parent)
        self.fig.patch.set_linewidth(5)
        self.fig.patch.set_edgecolor("black")
        FigureCanvas.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)

        self.app_data.raw_df_pd = self.app_data.raw_df.to_pandas()

        dsshow(
            app_data.raw_df_pd,
            ds.Point(self.app_data.params.x_col, self.app_data.params.y_col),
            norm="log",
            aspect="auto",
            ax=self.ax,
            cmap=cmr.horizon_r,
        )

        self.draw()


class LinCanvas(FigureCanvas):
    def __init__(self, app_data, parent=None, width=8, height=12, dpi=100, arg=0):
        self.arg = arg
        self.fig = plt.figure()
        FigureCanvas.__init__(self, self.fig)
        self.app_data = app_data
        self.fig.set_layout_engine("tight")
        # self.fig = Figure(figsize=(width, height), dpi=dpi)

        # self.fig, self.axs = plt.subplots(
        # 3,
        # 1,
        # gridspec_kw=dict(height_ratios=[3, 1, 1], wspace=0),
        # sharex=True,
        # )

        # self.fig.subplots_adjust(hspace=0)
        # self.ax1 = self.axs[0]
        # self.ax2 = self.axs[1]
        # self.ax3 = self.axs[2]

        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], figure=self.fig, hspace=0)
        self.gs = gs
        self.ax1 = host_subplot(gs[0], axes_class=HostAxes)
        self.ax2 = host_subplot(gs[1], axes_class=AA.Axes)
        self.ax3 = host_subplot(gs[2], axes_class=AA.Axes)

        # self.a_ax = self.ax1.twiny()
        # self.a_ax = self.ax1.get_aux_axes(sharex=self.ax1, viewlim_mode=None)
        # self.a_ax.sharex(self.ax1)
        # self.a_ax.axis["a"] = self.a_ax.new_fixed_axis(loc="top", offset=(0, 0))

        # self.z_ax = self.ax1.
        self.pid_ax = self.ax1.get_aux_axes(sharex=self.ax1, viewlim_mode=None)
        self.pid_ax.sharex(self.ax1)
        self.pid_ax.axis["pid"] = self.ax1.new_fixed_axis(loc="top", offset=(0, 0))

        self.app_data = app_data
        self.app_data.lin_ax = self.ax1
        self.app_data.lin_fig = self.fig
        self.app_data.pid_ax = self.pid_ax

        self.canvas = self.fig.canvas
        self.app_data.lin_canvas = self.canvas
        self.setParent(parent)
        self.fig.patch.set_linewidth(5)
        self.fig.patch.set_edgecolor("black")
        FigureCanvas.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)

        self.app_data.raw_df_pd = self.app_data.raw_df.to_pandas()
        x_col = self.app_data.params.x_col
        ad = self.app_data

        y_range = (ad.raw_df[x_col].min(), ad.raw_df[x_col].max())
        y_range_scale = y_range[1] - y_range[0]
        y_range = (y_range[0] - y_range_scale * 0.1, y_range[1] + y_range_scale * 0.1)
        dsshow(
            app_data.raw_df_pd,
            ds.Point("z_lin", self.app_data.params.x_col),
            norm="log",
            aspect="auto",
            ax=self.ax1,
            cmap=cmr.horizon_r,
            y_range=y_range,
        )
        self.ax2.hist(
            self.app_data.raw_df["z_lin"],
            histtype="step",
            color="w",
            bins=self.app_data.bins,
            fill="w",
        )
        self.ax3.hist(
            self.app_data.raw_df["z_lin"],
            histtype="step",
            color="w",
            bins=self.app_data.bins,
            fill="w",
        )
        self.ax3.set_yscale("log")

        self.ax3.set_xlabel("lin value")

        self.ax3.set_ylabel("counts")
        self.ax2.set_ylabel("counts")
        self.ax1.set_ylabel(self.app_data.params.x_col)

        self.canvas.mpl_connect("button_press_event", self.button_press_callback)
        self.canvas.mpl_connect("motion_notify_event", self.motion_notify_callback)

        self.app_data.update_pid_axes()

        self.draw()

    def motion_notify_callback(self, event):
        if event.inaxes is None:
            return
        ad = self.app_data
        match ad.mode:
            case AppMode.GATE_EDIT:
                return
            case _:
                pass

        nearest_non_active = ad.nearest_non_active_gate(event.xdata)
        if ad.active_gate is None:
            ad.active_gate = nearest_non_active
            nearest_non_active.interactor.update_visuals()
            ad.update_canvases()
            return

        if np.abs(ad.active_gate.x_center() - event.xdata) > np.abs(
            nearest_non_active.x_center() - event.xdata
        ):
            old_active = ad.active_gate
            ad.active_gate = nearest_non_active

            nearest_non_active.interactor.update_visuals()
            old_active.interactor.update_visuals()
            ad.update_canvases()

        if self.app_data.side_panel is not None:
            self.app_data.side_panel.pid_z_adjuster.set_label()
            self.app_data.side_panel.pid_a_adjuster.set_label()

        return

    def button_press_callback(self, event):
        if event.inaxes is None:
            return

        ad = self.app_data
        match ad.mode:
            case AppMode.GATE_SELECT:
                ad.mode = AppMode.GATE_EDIT
                ad.update_mode_selector()
                return
            case _:
                pass
        if ad.lin_nav_bar.mode in [_Mode.ZOOM, _Mode.PAN]:
            return

        if ad.active_gate is None:
            # ad.gates.append
            # ad.active_gate =
            reference_gate = ad.nearest_non_active_gate(event.xdata)
            width = None
            low = None
            high = None
            z = None
            a = None

            if reference_gate is None:
                width = (ad.raw_df["z_lin"].max() - ad.raw_df["z_lin"].min()) / 40
                low = ad.raw_df[ad.params.x_col].min()
                high = ad.raw_df[ad.params.x_col].max()
                z = 1
                a = 1
            else:
                width = reference_gate.right - reference_gate.left
                low = reference_gate.low
                high = reference_gate.high

                if reference_gate.x_center() < event.xdata:
                    a = reference_gate.a_lbl + 1

                else:
                    a = reference_gate.a_lbl - 1
                z = reference_gate.z_lbl

            match event.button:
                case 1:  # left click
                    new_gate = Gate(
                        z_lbl=z,
                        a_lbl=a,
                        low=low,
                        high=high,
                        left=event.xdata,
                        right=event.xdata + width,
                    )
                    new_gate.interactor = LinGateInteractor(ad.lin_ax, new_gate, ad)
                    ad.lin_interactor_manager.interactors.append(new_gate.interactor)

                    ad.gates.append(new_gate)
                    ad.active_gate = new_gate
                    new_gate.interactor.update_visuals()
                    # ad.lin()
                    self.app_data.update_pid_axes()
                    ad.update_canvases()
                    return

                case 3:  # right click
                    new_gate = Gate(
                        z_lbl=z,
                        a_lbl=a,
                        low=low,
                        high=high,
                        left=event.xdata,
                        right=event.xdata + width,
                    )
                    ad.gates.append(new_gate)
                    ad.active_gate = new_gate
                    ad.update_interactors()
                case _:
                    pass

            self.app_data.update_pid_axes()
            self.canvas.draw()
            return

        match event.button:
            case 1:  # left click
                if ad.vertical_mode:
                    ad.active_gate.set_bottom(event.ydata)
                else:
                    ad.active_gate.set_left(event.xdata)
            case 3:  # right click
                if ad.vertical_mode:
                    ad.active_gate.set_top(event.ydata)
                else:
                    ad.active_gate.set_right(event.xdata)
            case _:
                pass

        if ad.active_gate.interactor is not None:
            ad.active_gate.interactor.update_visuals()
            self.update_axes()
        else:
            self.app_data.update_interactors()
        self.app_data.update_pid_axes()
        self.canvas.draw()

    def update_axes(self):
        return


class PidAdjuster(QWidget):
    def __init__(self, app_data, lbl):
        super().__init__()
        self.app_data = app_data
        self.lbl = lbl
        self.layout = QHBoxLayout(self)
        self.increment_btn = QPushButton("+")
        self.big_increment_btn = QPushButton("++")
        self.big_decrement_btn = QPushButton("--")
        self.decrement_btn = QPushButton("-")
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.big_decrement_btn)
        self.layout.addWidget(self.decrement_btn)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.increment_btn)
        self.layout.addWidget(self.big_increment_btn)

    def gate(self):
        return self.app_data.active_gate

    def set_label(self):
        if self.gate() is not None:
            self.label.setText(f"{self.lbl}")

    def small_inc(self):
        return

    def big_inc(self):
        return

    def big_dec(self):
        return

    def small_dec(self):
        return


class PidZAdjuster(QWidget):
    def __init__(self, app_data, lbl):
        super().__init__()
        self.app_data = app_data
        self.lbl = lbl
        self.layout = QHBoxLayout(self)
        self.increment_btn = QPushButton("+ (2)")
        self.big_increment_btn = QPushButton("++")
        self.big_decrement_btn = QPushButton("--")
        self.decrement_btn = QPushButton("- (1)")
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.big_decrement_btn)
        self.layout.addWidget(self.decrement_btn)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.increment_btn)
        self.layout.addWidget(self.big_increment_btn)

        self.increment_btn.clicked.connect(self.small_inc)
        self.big_increment_btn.clicked.connect(self.big_inc)
        self.decrement_btn.clicked.connect(self.small_dec)
        self.big_decrement_btn.clicked.connect(self.big_dec)

        self.set_label()

    def gate(self):
        return self.app_data.active_gate

    def set_label(self):
        if self.gate() is not None:
            self.label.setText(f"{self.lbl}: {self.gate().z_lbl}")
        else:
            self.label.setText(f"{self.lbl}")
        return

    def small_inc(self):
        if self.gate() is not None:
            self.gate().z_lbl += 1
            self.gate().interactor.update_visuals()
            self.app_data.update_pid_axes()
            self.app_data.update_canvases()
            self.set_label()


        return

    def big_inc(self):
        if self.gate() is not None:
            self.gate().z_lbl += 3
            self.gate().interactor.update_visuals()
            self.app_data.update_pid_axes()
            self.app_data.update_canvases()
            self.set_label()
        return

    def big_dec(self):
        if self.gate() is not None:
            self.gate().z_lbl -= 3
            self.gate().interactor.update_visuals()
            self.app_data.update_pid_axes()
            self.app_data.update_canvases()
            self.set_label()
        return

    def small_dec(self):
        if self.gate() is not None:
            self.gate().z_lbl -= 1
            self.gate().interactor.update_visuals()
            self.app_data.update_pid_axes()
            self.app_data.update_canvases()
            self.set_label()
        return


class PidAAdjuster(QWidget):
    def __init__(self, app_data, lbl):
        super().__init__()
        self.app_data = app_data
        self.lbl = lbl
        self.layout = QHBoxLayout(self)
        self.increment_btn = QPushButton("+ (4)")
        self.big_increment_btn = QPushButton("++")
        self.big_decrement_btn = QPushButton("--")
        self.decrement_btn = QPushButton("- (3)")
        self.label = QLabel("")
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.big_decrement_btn)
        self.layout.addWidget(self.decrement_btn)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.increment_btn)
        self.layout.addWidget(self.big_increment_btn)

        self.increment_btn.clicked.connect(self.small_inc)
        self.big_increment_btn.clicked.connect(self.big_inc)
        self.decrement_btn.clicked.connect(self.small_dec)
        self.big_decrement_btn.clicked.connect(self.big_dec)

        self.set_label()

    def gate(self):
        return self.app_data.active_gate

    def set_label(self):
        if self.gate() is not None:
            self.label.setText(f"{self.lbl}: {self.gate().a_lbl}")
        else:
            self.label.setText(f"{self.lbl}")
        return

    def small_inc(self):
        if self.gate() is not None:
            self.gate().a_lbl += 1
            self.gate().interactor.update_visuals()
            self.app_data.update_pid_axes()
            self.app_data.update_canvases()
            self.set_label()

        return

    def big_inc(self):
        if self.gate() is not None:
            self.gate().a_lbl += 3
            self.gate().interactor.update_visuals()
            self.app_data.update_pid_axes()
            self.app_data.update_canvases()
            self.set_label()
        return

    def big_dec(self):
        if self.gate() is not None:
            self.gate().a_lbl -= 3
            self.gate().interactor.update_visuals()
            self.app_data.update_pid_axes()
            self.app_data.update_canvases()
            self.set_label()
        return

    def small_dec(self):
        if self.gate() is not None:
            self.gate().a_lbl -= 1
            self.gate().interactor.update_visuals()
            self.app_data.update_pid_axes()
            self.app_data.update_canvases()
            self.set_label()
        return


class ImportExportPanel(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QHBoxLayout(self)
        self.export_btn = QPushButton("Export")
        self.import_btn = QPushButton("Import")
        self.layout.addWidget(self.export_btn, alignment=Qt.AlignTop)
        self.layout.addWidget(self.import_btn, alignment=Qt.AlignTop)

        self.export_btn.clicked.connect(self.export_gates)
        self.import_btn.clicked.connect(self.import_gates)

    def export_gates(self):
        print("exporting ...")
        ad = self.app_data
        saved_gates = ad.saved_gates()

        file_name = os.path.join(ad.directory, "limits.dat")

        with open(file_name, "w") as output_file:
            for line_idx, g in enumerate(saved_gates):
                if line_idx != 0:
                    output_file.write("\n")
                output_file.write(
                    f"{g.z_lbl} {g.a_lbl} {g.left} {g.right} {g.low} {g.high}"
                )
        print("...done")
        return

    def import_gates(self):
        print("importing ...")
        ad = self.app_data

        file_name = os.path.join(ad.directory, "limits.dat")

        gates = []
        with open(file_name, "r") as input_file:
            for line in input_file:
                words = line.split(" ")
                for idx, w in enumerate(words):
                    w = w.strip()
                new_gate = Gate(
                    z_lbl=int(words[0]),
                    a_lbl=int(words[1]),
                    left=float(words[2]),
                    right=float(words[3]),
                    low=float(words[4]),
                    high=float(words[5]),
                )
                gates.append(new_gate)
        ad.gates = gates
        ad.active_gate = None
        self.app_data.update_pid_axes()
        ad.update_interactors()
        print("...done")
        return


class ModeSelector(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.app_data.mode_selector = self
        self.layout = QVBoxLayout(self)

        self.group_box = QGroupBox("App Mode")
        self.group_box_layout = QHBoxLayout()
        self.group_box.setLayout(self.group_box_layout)

        self.layout.addWidget(self.group_box, alignment=QtCore.Qt.AlignTop)

        self.edit_mode_radio_btn = QRadioButton("Edit (E)")
        # self.delete_mode_radio_btn = QRadioButton("Delete (D)")
        self.select_gate_radio_btn = QRadioButton("Gate Select (G)")

        self.edit_mode_radio_btn.toggled.connect(self.app_data.set_mode_to_edit)

        # self.delete_mode_radio_btn.toggled.connect(self.app_data.set_mode_to_delete)

        self.select_gate_radio_btn.toggled.connect(
            self.app_data.set_mode_to_gate_select
        )

        self.group_box_layout.addWidget(
            self.edit_mode_radio_btn, alignment=QtCore.Qt.AlignTop
        )
        # self.group_box_layout.addWidget(
        # self.delete_mode_radio_btn, alignment=QtCore.Qt.AlignTop
        # )
        self.group_box_layout.addWidget(
            self.select_gate_radio_btn, alignment=QtCore.Qt.AlignTop
        )

        self.update_radio_btn()
        self.setLayout(self.layout)

    def update_radio_btn(self):
        radio_btn = None
        match self.app_data.mode:
            case AppMode.GATE_EDIT:
                radio_btn = self.edit_mode_radio_btn
            # case AppMode.DELETE:
            # radio_btn = self.delete_mode_radio_btn
            case AppMode.GATE_SELECT:
                radio_btn = self.select_gate_radio_btn

        if not radio_btn.isChecked():
            radio_btn.toggle()

    # def delete_mode(self):
    # self.app_data.mode = AppMode.DELETE
    # self.update_radio_btn()

    def gate_select_mode(self):
        self.app_data.mode = AppMode.GATE_SELECT
        self.update_radio_btn()


class SidePanel(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QVBoxLayout(self)
        self.raw_panel = RawPanel(app_data)
        self.pid_z_adjuster = PidZAdjuster(app_data, "Z")
        self.pid_a_adjuster = PidAAdjuster(app_data, "A")

        self.import_export_panel = ImportExportPanel(app_data)
        self.mode_selector = ModeSelector(app_data)
        self.app_data.side_panel = self

        QWidget.setSizePolicy(
            self.pid_z_adjuster,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        QWidget.setSizePolicy(
            self.pid_a_adjuster,
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        QWidget.updateGeometry(self)

        self.layout.addWidget(self.pid_z_adjuster, alignment=Qt.AlignTop)
        self.layout.addWidget(self.pid_a_adjuster, alignment=Qt.AlignTop)
        self.layout.addWidget(self.import_export_panel, alignment=Qt.AlignTop)
        self.layout.addWidget(self.mode_selector, alignment=Qt.AlignTop)
        self.layout.addWidget(self.raw_panel)
        self.setLayout(self.layout)


class MakeLimitsGUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.left = 100
        self.top = 100
        self.title = "Point Picking GUI"
        self.width = 1200
        self.height = 1600

        self.app_data = AppData()

        wid = QWidget()
        self.setCentralWidget(wid)
        main_layout = QHBoxLayout(wid)
        self.side_panel = SidePanel(app_data=self.app_data)
        self.lin_panel = LinPanel(app_data=self.app_data)
        main_layout.addWidget(self.side_panel, stretch=1)
        main_layout.addWidget(self.lin_panel, stretch=2)

        self.app_data.update_interactors()

        self.show()

    def delete_gate(self):
        ad = self.app_data
        if ad.active_gate is None:
            return

        interactors = [
            interactor
            for interactor in ad.lin_interactor_manager.interactors
            if interactor != ad.active_gate.interactor
        ]
        gates = [gate for gate in ad.gates if gate != ad.active_gate]

        self.app_data.gates = gates
        self.app_data.interactors = interactors
        ad.active_gate = None
        self.app_data.update_interactors()

    def save_gate(self):
        ad = self.app_data
        active_gate = ad.active_gate
        new_gate = Gate(
            z_lbl=active_gate.z_lbl,
            a_lbl=active_gate.a_lbl + 1,
            high=active_gate.high,
            low=active_gate.low,
            left=active_gate.right,
            right=2 * active_gate.right - active_gate.left,
        )
        ad.gates.append(new_gate)
        ad.active_gate = new_gate
        new_gate.interactor = LinGateInteractor(ad.lin_ax, new_gate, ad)
        ad.lin_interactor_manager.interactors.append(new_gate.interactor)

        new_gate.interactor.update_visuals()
        active_gate.interactor.update_visuals()  # old active gate
        # ad.gates[len(ad.gates) - 2].interactor.update_visuals()

        ad.update_pid_axes()
        ad.update_canvases()

        self.side_panel.pid_z_adjuster.set_label()
        self.side_panel.pid_a_adjuster.set_label()

    def keyPressEvent(self, e):
        if e.isAutoRepeat():
            return
        if e.key() == Qt.Key_V:
            self.app_data.vertical_mode = True
        if e.key() == Qt.Key_S:
            self.save_gate()
        if e.key() == Qt.Key_D:
            self.delete_gate()
        if e.key() == Qt.Key_E:
            self.app_data.mode = AppMode.GATE_EDIT
            self.app_data.update_mode_selector()
        if e.key() == Qt.Key_G:
            self.app_data.mode = AppMode.GATE_SELECT
            self.app_data.update_mode_selector()
        if e.key() == Qt.Key_T:
            self.app_data.lin_canvas
            if self.app_data.lin_nav_bar.mode == _Mode.ZOOM:
                self.app_data.lin_nav_bar.zoom()
            elif self.app_data.lin_nav_bar.mode == _Mode.PAN:
                self.app_data.lin_nav_bar.pan()
            else:
                self.app_data.lin_nav_bar.zoom()
        if e.key() == Qt.Key_1:
            self.app_data.side_panel.pid_z_adjuster.small_dec()
        if e.key() == Qt.Key_2:
            self.app_data.side_panel.pid_z_adjuster.small_inc()
        if e.key() == Qt.Key_3:
            self.app_data.side_panel.pid_a_adjuster.small_dec()
        if e.key() == Qt.Key_4:
            self.app_data.side_panel.pid_a_adjuster.small_inc()

    def keyReleaseEvent(self, e):
        if e.isAutoRepeat():
            return
        if e.key() == Qt.Key_V:
            self.app_data.vertical_mode = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Force the style to be the same on all OSs:
    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(65, 65, 65))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(65, 65, 65))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(65, 65, 65))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    ex = MakeLimitsGUI()
    sys.exit(app.exec_())
