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
        self.label = Text(
            0.5 * (self.gate.left + self.gate.right),
            self.gate.low - (self.gate.high - self.gate.low) * 0.05,
            f"({self.gate.z_lbl}, {self.gate.a_lbl})",
            verticalalignment="top",
            horizontalalignment="center",
            animated=True,
            rotation=90,
            bbox=dict(facecolor=(0.2, 0.2, 0.2, 0.8)),
        )

        self.ax.add_patch(self.rect)

    def update_visuals(self):
        self.rect.set_bounds(
            self.gate.left,
            self.gate.low,
            self.gate.right - self.gate.left,
            self.gate.high - self.gate.low,
        )
        self.label.set_x(
            0.5 * (self.gate.left + self.gate.right),
        )
        self.label.set_y(
            self.gate.low - (self.gate.high - self.gate.low) * 0.05,
        )

    def draw_callback(self, event):
        self.ax.add_patch(self.rect)
        self.ax.add_artist(self.label)
        self.ax.draw_artist(self.rect)
        self.ax.draw_artist(self.label)
        return

    def motion_notify_callback(self, event):
        return

    def clean(self):
        self.rect.remove()
        self.label.remove()
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
        self.out_file = "lims.dat"
        self.bins = args.bins
        self.params = ParamManager(self.directory)

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

        test_gate = Gate()
        test_gate.low = 5
        test_gate.high = 50
        test_gate.left = 39.2
        test_gate.right = 43
        test_gate.z_lbl = 1
        test_gate.a_lbl = 1
        test_gate.locked = False
        self.gates = [test_gate]
        self.lin_gate_interactors = []
        self.lin_interactor_manager = None

        self.active_gate = test_gate

        self.vertical_mode = False

        self.lin_nav_bar = None
        return

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
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        self.fig, self.axs = plt.subplots(
            3,
            1,
            gridspec_kw=dict(height_ratios=[3, 1, 1], wspace=0),
            sharex=True,
        )

        self.fig.subplots_adjust(hspace=0)
        self.ax1 = self.axs[0]
        self.ax2 = self.axs[1]
        self.ax3 = self.axs[2]
        self.app_data = app_data
        self.app_data.lin_ax = self.ax1
        self.app_data.lin_fig = self.fig
        FigureCanvas.__init__(self, self.fig)
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
        y_range = (y_range[0] - y_range_scale * 0.3, y_range[1] + y_range_scale * 0.1)
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

        self.draw()

    def button_press_callback(self, event):
        if event.inaxes is None:
            return

        ad = self.app_data

        if ad.active_gate is None:
            return

        if ad.lin_nav_bar.mode in [_Mode.ZOOM, _Mode.PAN]:
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
        self.canvas.draw()

    def update_axes(self):
        return


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
        self.raw_panel = RawPanel(app_data=self.app_data)
        self.lin_panel = LinPanel(app_data=self.app_data)
        main_layout.addWidget(self.raw_panel, stretch=1)
        main_layout.addWidget(self.lin_panel, stretch=2)

        delete_shortcut = QKeySequence(Qt.Key_D)
        self.delete_shortcut = QShortcut(delete_shortcut, self)
        self.delete_shortcut.activated.connect(self.delete_data)

        self.app_data.update_interactors()

        self.show()

    def delete_data(self):
        self.app_data.gates = []
        self.app_data.update_interactors()
        self.app_data.update_canvases()

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
        ad.update_interactors()

    def keyPressEvent(self, e):
        if e.isAutoRepeat():
            return
        if e.key() == Qt.Key_V:
            self.app_data.vertical_mode = True
        if e.key() == Qt.Key_S:
            self.save_gate()

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

    import sys
