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


class Gate:
    def __init__(self):
        self.z_lbl = 0
        self.a_lbl = 0

        self.low = 0
        self.high = 0
        self.left = 0
        self.right = 0


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

        raw_df_file = os.path.join(self.directory, self.raw_file)

        lf = pl.scan_parquet(raw_df_file).select(
            [self.params.x_col, self.params.y_col, "z_lin"]
        )
        if args.limit is not None:
            lf = lf.limit(args.limit)

        self.raw_df = lf.collect()
        self.raw_df_pd = self.raw_df.to_pandas

        return


class RawPanel(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QVBoxLayout(self)

        main_plot = LinCanvas(self.app_data, self, width=8, height=6)

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
        # self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig, self.axs = plt.subplots(
            3,
            1,
            # gridspec_kw=dict(height_ratios=[3, 1, 1], wspace=0),
            # gridspec_kw=dict(height_ratios=[3, 1, 1], wspace=0),
        )
        self.arg = arg
        self.ax1 = self.axs[0]
        self.ax2 = self.axs[1]
        self.ax3 = self.axs[3]

        # self.ax1 = self.fig.add_subplot(5, 1, (1,3))
        # self.ax2 = self.fig.add_subplot(5, 1, 4)
        # self.ax3 = self.fig.add_subplot(5, 1, 5)
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
            ax=self.ax1,
            # cmap=cmr.neutral,
            cmap=cmr.ocean_r,
            # alpha=0.5,
        )
        self.ax2.hist(
            self.app_data.raw_df["z_lin"],
            histtype="step",
            color="white",
            bins=self.app_data.bins,
        )
        self.ax3.hist(
            self.app_data.raw_df["z_lin"],
            histtype="step",
            color="white",
            bins=self.app_data.bins,
        )
        self.ax3.set_yscale("log")
        self.ax3.sharex(self.ax1)
        self.ax2.sharex(self.ax1)

        self.draw()

        # self.layout = QVBoxLayout(self)


class LinCanvas(FigureCanvas):
    def __init__(self, app_data, parent=None, width=8, height=6, dpi=100, arg=0):
        self.arg = arg
        self.fig, self.axs = plt.subplots(
            3, 1, gridspec_kw=dict(height_ratios=[3, 1, 1], wspace=0), sharex=True
        )
        self.fig.subplots_adjust(hspace=0)
        self.arg = arg
        self.ax1 = self.axs[0]
        self.ax2 = self.axs[1]
        self.ax3 = self.axs[2]
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
        )
        self.ax3.hist(
            self.app_data.raw_df["z_lin"],
            histtype="step",
            color="w",
            bins=self.app_data.bins,
        )
        self.ax3.set_yscale("log")
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
        self.lin_panel = LinPanel(app_data=self.app_data)
        main_layout.addWidget(self.lin_panel)

        self.show()


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
