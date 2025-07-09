import argparse
import copy
import os
import sys
from enum import Enum

import cmasher as cmr
import datashader as ds
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from datashader.mpl_ext import dsshow
from matplotlib.backend_bases import _Mode
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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
    def __init__(self):
        self.z_lbl = 0
        self.a_lbl = 0

        self.low = 0
        self.high = 0
        self.left = 0
        self.right = 0

        self.locked = True


class LinGateInteractor(QObject):
    def __init__(self, ax, gate, app_data):
        super().__init__()
        if ax is None:
            return
        self.gate = gate
        self.app_data = app_data
        self.ax = ax

        if ax is not None:
            self.canvas = ax.figure.canvas

        self.rect = Rectangle(
            [gate.left, gate.low],
            gate.right - gate.left,
            gate.high - gate.low,
            facecolor=(0.3, 0, 0, 0.3),
            edgecolor="red",
            lw=0.5,
        )
        # self.rect = Line2D(
        # [gate.left, gate.right, gate.right, gate.left, gate.left],
        # [gate.low, gate.low, gate.high, gate.high, gate.low],
        # marker="",
        # color="red",
        # animated=True,
        # )
        if self.ax is not None:
            print("add_patch")
            # self.ax.annotate(
            # f"({self.gate.z_lbl}, {self.gate.a_lbl})",
            # xy=(
            # 0.5 * (self.gate.left + self.gate.right),
            # self.gate.low + (self.gate.high - self.gate.low) * 1.05,
            # ),
            # xycoords="data",
            # ha="center",
            # va="bottom",
            # rotation=90,
            # )
            # self.ax.add_line(self.rect)

    def update_visuals(self):
        print(self.gate.left)
        print(self.gate.right)
        self.rect = Rectangle(
            [self.gate.left, self.gate.low],
            self.gate.right - self.gate.left,
            self.gate.high - self.gate.low,
            facecolor=(0.3, 0, 0, 0.3),
            edgecolor="red",
            lw=0.5,
        )
        # self.draw()

    def draw_callback(self, event):
        # if self.ax is not None:
        self.ax.add_patch(self.rect)
        self.ax.draw_artist(self.rect)
        return

    def motion_notify_callback(self, event):
        if self.gate.locked:
            return
        half_width = (self.gate.right - self.gate.left) / 2
        self.gate.left = event.xdata - half_width
        self.gate.right = event.xdata + half_width
        self.update_visuals()
        return


class InteractorManager(QObject):
    def __init__(self, fig, ax, interactors):
        super().__init__()

        self.interactors = interactors
        self.ax = ax
        self.fig = fig
        self.canvas = self.fig.canvas
        self.canvas.mpl_connect("draw_event", self.draw_callback)
        self.canvas.mpl_connect("motion_notify_event", self.motion_notify_callback)
        # self.canvas.mpl_connect("scroll_event", self.update_bg_callback)
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

        test_gate = Gate()
        test_gate.low = 2
        test_gate.high = 50
        test_gate.left = 39.2
        test_gate.right = 43
        test_gate.z_lbl = 2
        test_gate.a_lbl = 4
        test_gate.locked = False
        self.gates = [test_gate]
        self.lin_gate_interactors = []
        self.lin_interactor_manager = None

        return

    def update_interactors(self):
        if self.lin_ax is not None:
            # self.lin_gate_interactors = [
            # LinGateInteractor(self.lin_ax, g, self) for g in self.gates
            # ]

            if self.lin_interactor_manager is not None:
                print(len(self.lin_interactor_manager.interactors))
            self.lin_interactor_manager = InteractorManager(
                self.lin_fig,
                self.lin_ax,
                [LinGateInteractor(self.lin_ax, g, self) for g in self.gates],
            )

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
        self.app_data.nav_bar = NavigationToolbar(main_plot.canvas, self)
        self.layout.addWidget(self.app_data.nav_bar)

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

        # self.layout = QVBoxLayout(self)


class LinCanvas(FigureCanvas):
    def __init__(self, app_data, parent=None, width=8, height=12, dpi=100, arg=0):
        self.arg = arg
        self.fig, self.axs = plt.subplots(
            3, 1, gridspec_kw=dict(height_ratios=[3, 1, 1], wspace=0), sharex=True
        )
        # self.fig = Figure(figsize=(width, height), dpi=dpi)
        # self.ax1 = self.fig.axes
        # self.ax1 = self.fig.add_subplot(111)

        self.fig.subplots_adjust(hspace=0)
        self.arg = arg
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
        dsshow(
            app_data.raw_df_pd,
            ds.Point("z_lin", self.app_data.params.x_col),
            norm="log",
            aspect="auto",
            ax=self.ax1,
            cmap=cmr.horizon_r,
            # cmap=cmr.rainforest_r,
            # cmap=cmr.tropical,
            # alpha=0.5,
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
        # self.ax3.sharex(self.ax1)
        # self.ax2.sharex(self.ax1)

        self.draw()


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
