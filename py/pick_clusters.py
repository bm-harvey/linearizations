#!/usr/bin/env python
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


class CurveStatus(Enum):
    ACTIVE = 0
    INACTIVE = 1


class AppMode(Enum):
    NORMAL = 0
    POINT_PICKING = 1
    POINT_DELETE = 2
    CURVE_SELECT = 3


class AppData:
    def __init__(self, gui):
        # app modes
        self.mode = AppMode.NORMAL

        # settings
        self.inactive_color = "blue"
        self.active_color = "red"

        # stored data
        self.saved_curves = []
        self.active_curve = None
        self.ridges_df = None

        self.nav_bar = None
        self.gui = gui
        parser = argparse.ArgumentParser()
        parser.add_argument("-d", "--directory", default="data/faust_det_60")
        parser.add_argument("-i", "--input_file", default="ridges.parquet")
        parser.add_argument("-r", "--raw_file", default="raw.parquet")
        parser.add_argument("-o", "--output_file", default="clustered_points.parquet")
        parser.add_argument("-c", "--curve_file", default="curves.parquet")

        args = parser.parse_args()
        param_manager = ParamManager(args.directory)

        self.directory = args.directory
        self.in_file = args.input_file
        self.raw_file = args.raw_file
        self.out_file = args.output_file
        self.curve_file = args.curve_file
        self.x_col = param_manager.x_col
        self.y_col = param_manager.y_col

    def get_color(self, status):
        match status:
            case CurveStatus.INACTIVE:
                return self.inactive_color
            case CurveStatus.ACTIVE:
                return self.active_color
            case _:
                return "white"

    def set_mode_to_normal(self):
        self.mode = AppMode.NORMAL
        if self.nav_bar is not None:
            if self.nav_bar.mode != _Mode.ZOOM:
                self.nav_bar.zoom()

    def set_mode_to_point_picking(self):
        self.mode = AppMode.POINT_PICKING
        if self.nav_bar is not None:
            if self.nav_bar.mode == _Mode.ZOOM:
                self.nav_bar.zoom()
            if self.nav_bar.mode == _Mode.PAN:
                self.nav_bar.pan()

    def set_mode_to_point_delete(self):
        self.mode = AppMode.POINT_DELETE
        if self.nav_bar is not None:
            if self.nav_bar.mode == _Mode.ZOOM:
                self.nav_bar.zoom()
            if self.nav_bar.mode == _Mode.PAN:
                self.nav_bar.pan()

    def set_mode_to_curve_selection(self):
        self.mode = AppMode.CURVE_SELECT
        if self.nav_bar is not None:
            if self.nav_bar.mode == _Mode.ZOOM:
                self.nav_bar.zoom()
            if self.nav_bar.mode == _Mode.PAN:
                self.nav_bar.pan()


class Curve:
    def __init__(self, xs=[], ys=[], window_size=1):
        self.xs = xs
        self.ys = ys
        self.window_size = window_size

    def set_window_size(self, window_size):
        self.window_size = window_size

    def add_point(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        self.sort()

    def remove_point_by_idx(self, idx):
        self.xs.pop(idx)
        self.ys.pop(idx)

    def sort(self):
        self.xs, self.ys = (list(t) for t in zip(*sorted(zip(self.xs, self.ys))))

    def len(self) -> int:
        return len(self.xs)

    def clear(self):
        self.xs = []
        self.ys = []

    def print_points(self):
        print("x: ")
        print(self.xs)
        print("y: ")
        print(self.ys)

    def point_is_in_curve(self, x, y):
        for idx in range(self.len() - 1):
            if x < self.xs[idx]:
                continue
            if x > self.xs[idx + 1]:
                continue
            slope = (self.ys[idx + 1] - self.ys[idx]) / (
                self.xs[idx + 1] - self.xs[idx]
            )

            y_curve = slope * (x - self.xs[idx]) + self.ys[idx]

            if (
                y_curve + self.window_size / 2 > y
                and y_curve - self.window_size / 2 < y
            ):
                return True
            return False


class CurveInteractor(QObject):
    def __init__(
        self, ax, xs, ys, app_data, status=CurveStatus.INACTIVE, window_size=None
    ):
        super().__init__()
        self.app_data = app_data
        self.status = status

        self.curve = Curve(xs, ys)
        if window_size is None:
            try:
                window_size = float(
                    self.app_data.window_size_manager.window_size_txt.text()
                )
                self.curve.window_size = window_size
            except ValueError:
                pass
        else:
            self.curve.window_size = window_size

        self.ax = ax
        self.canvas = ax.figure.canvas

        color = self.app_data.inactive_color
        if self.status == CurveStatus.ACTIVE:
            color = self.app_data.active_color

        self.line_low = Line2D(
            self.curve.xs,
            self.curve.ys,
            marker="^",
            markerfacecolor=color,
            color=color,
            animated=True,
        )
        self.line_mid = Line2D(
            self.curve.xs,
            self.curve.ys,
            marker="+",
            markerfacecolor=color,
            color=color,
            animated=True,
            ls="-",
            lw=0.5,
        )
        self.line_high = Line2D(
            self.curve.xs,
            self.curve.ys,
            marker="v",
            markerfacecolor=color,
            color=color,
            animated=True,
        )

        self.draw_virtual_line_right = False
        self.draw_virtual_line_left = False

        self.point_to_delete = Line2D(
            [0, 0],
            [0, 0],
            marker="x",
            markerfacecolor=color,
            color=color,
            ls=":",
            alpha=0.5,
            ms=25,
            animated=True,
        )
        self.virtual_line_right = Line2D(
            [0, 0],
            [0, 0],
            marker="o",
            markerfacecolor=color,
            color=color,
            ls=":",
            alpha=0.5,
            animated=True,
        )
        self.point_to_mouse_line = Line2D(
            [0, 0],
            [0, 0],
            marker="",
            markerfacecolor=self.app_data.active_color,
            color=self.app_data.active_color,
            ls=":",
            alpha=0.5,
            animated=True,
        )
        self.virtual_line_left = Line2D(
            [0, 0],
            [0, 0],
            marker="o",
            markerfacecolor=color,
            color=color,
            ls=":",
            alpha=0.5,
            animated=True,
        )
        self.ax.add_line(self.line_low)
        self.ax.add_line(self.line_high)
        self.ax.add_line(self.line_mid)
        self.ax.add_line(self.virtual_line_right)
        self.ax.add_line(self.virtual_line_left)
        self.ax.add_line(self.point_to_mouse_line)
        self.ax.add_line(self.point_to_delete)

        self._ind = None  # the active vert

        self.canvas.mpl_connect("button_press_event", self.button_press_callback)

    def draw_callback(self, event):
        self.ax.draw_artist(self.line_mid)
        if self.status == CurveStatus.INACTIVE:
            return
        self.ax.draw_artist(self.line_low)
        self.ax.draw_artist(self.line_high)
        match self.app_data.mode:
            case AppMode.POINT_PICKING:
                if self.draw_virtual_line_right:
                    self.ax.draw_artist(self.virtual_line_right)
                if self.draw_virtual_line_left:
                    self.ax.draw_artist(self.virtual_line_left)
            case AppMode.POINT_DELETE:
                self.ax.draw_artist(self.point_to_delete)
            case AppMode.CURVE_SELECT:
                self.ax.draw_artist(self.point_to_mouse_line)

    def motion_notify_callback(self, event):
        "on mouse movement"
        if self.status == CurveStatus.INACTIVE:
            return
        self.draw_virtual_line_right = True
        match self.app_data.mode:
            case AppMode.POINT_PICKING:
                self.update_virtual_line(event.xdata, event.ydata)
            case AppMode.POINT_DELETE:
                self.update_point_delete(event.xdata, event.ydata)
            case AppMode.CURVE_SELECT:
                self.update_point_to_mouse(event.xdata, event.ydata)

    def get_dist_sqr_to_closest(self, x, y) -> int:
        min_dist_sqr = None
        if self.curve.len() == 0:
            return None
        for idx in range(self.curve.len()):
            pnt_x = self.curve.xs[idx]
            pnt_y = self.curve.ys[idx]

            dist_sqr = (pnt_x - x) ** 2 + (pnt_y - y) ** 2
            if min_dist_sqr is None or dist_sqr < min_dist_sqr:
                min_dist_sqr = dist_sqr
        return min_dist_sqr

    def get_idx_of_closest(self, x, y) -> int:
        closest_idx = 0
        min_dist_sqr = None
        if self.curve.len() == 0:
            return None
        for idx in range(self.curve.len()):
            pnt_x = self.curve.xs[idx]
            pnt_y = self.curve.ys[idx]

            dist_sqr = (pnt_x - x) ** 2 + (pnt_y - y) ** 2
            if min_dist_sqr is None or dist_sqr < min_dist_sqr:
                min_dist_sqr = dist_sqr
                closest_idx = idx
        return closest_idx

    def update_point_delete(self, x, y):
        if self.status == CurveStatus.INACTIVE:
            return
        closest_idx = self.get_idx_of_closest(x, y)
        if closest_idx is None:
            self.point_to_delete.set_data([], [])
        else:
            self.point_to_delete.set_data(
                [self.curve.xs[closest_idx]],
                [self.curve.ys[closest_idx]],
            )

    def update_point_to_mouse(self, x, y):
        if self.status == CurveStatus.INACTIVE:
            return
        closest_idx = self.get_idx_of_closest(x, y)
        if closest_idx is None:
            self.point_to_mouse_line.set_data([], [])
        else:
            self.point_to_mouse_line.set_data(
                [self.curve.xs[closest_idx], x],
                [self.curve.ys[closest_idx], y],
            )

    def update_virtual_line(self, x, y):
        if self.status == CurveStatus.INACTIVE:
            return
        idx_left = int(np.array(self.curve.xs).searchsorted(x, "right") - 1)

        if idx_left >= 0:
            self.virtual_line_right.set_data(
                [self.curve.xs[idx_left], x],
                [self.curve.ys[idx_left], y],
            )
            self.draw_virtual_line_right = True
        else:
            self.draw_virtual_line_right = False

        if idx_left < len(self.curve.xs) - 1:
            self.virtual_line_left.set_data(
                [x, self.curve.xs[idx_left + 1]],
                [y, self.curve.ys[idx_left + 1]],
            )
            self.draw_virtual_line_left = True
        else:
            self.draw_virtual_line_left = False

    def update_line(self, force_update=False):
        if self.status == CurveStatus.INACTIVE and not force_update:
            return
        self.line_low.set_data(
            self.curve.xs, [y - self.curve.window_size / 2 for y in self.curve.ys]
        )
        self.line_high.set_data(
            self.curve.xs, [y + self.curve.window_size / 2 for y in self.curve.ys]
        )
        self.line_mid.set_data(self.curve.xs, self.curve.ys)

    def set_status(self, status):
        if status == self.status:
            return

        self.status = status
        self.update_line_colors()
        self.app_data.canvas.draw_idle()

    def update_line_colors(self):
        self.line_low.set_color(self.app_data.get_color(self.status))
        self.line_low.set_markerfacecolor(self.app_data.get_color(self.status))
        self.line_mid.set_color(self.app_data.get_color(self.status))
        self.line_mid.set_markerfacecolor(self.app_data.get_color(self.status))
        self.line_high.set_color(self.app_data.get_color(self.status))
        self.line_high.set_markerfacecolor(self.app_data.get_color(self.status))

        self.virtual_line_right.set_color(self.app_data.get_color(self.status))
        self.virtual_line_right.set_markerfacecolor(
            self.app_data.get_color(self.status)
        )

        self.virtual_line_left.set_color(self.app_data.get_color(self.status))
        self.virtual_line_left.set_markerfacecolor(self.app_data.get_color(self.status))
        return

    def button_press_callback(self, event):
        "whenever a mouse button is pressed"
        if self.status == CurveStatus.INACTIVE:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        match self.app_data.mode:
            case AppMode.POINT_PICKING:
                self.curve.add_point(event.xdata, event.ydata)
                self.update_virtual_line(event.xdata, event.ydata)
                self.update_line()
            case AppMode.POINT_DELETE:
                delete_idx = self.get_idx_of_closest(event.xdata, event.ydata)
                if delete_idx is not None:
                    self.curve.remove_point_by_idx(delete_idx)
                    self.update_line()
                    self.update_point_delete(event.xdata, event.ydata)

        self.canvas.draw_idle()

    def clear(self):
        self.curve.clear()
        self.app_data.main_canvas.draw_idle()


class ModeSelector(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QVBoxLayout(self)

        self.group_box = QGroupBox("App Mode")
        self.group_box_layout = QVBoxLayout()
        self.group_box.setLayout(self.group_box_layout)

        self.layout.addWidget(self.group_box, alignment=QtCore.Qt.AlignTop)

        self.normal_mode_radio_btn = QRadioButton("Normal (T)")
        self.point_pick_mode_radio_btn = QRadioButton("Point Pick (T)")
        self.point_delete_mode_radio_btn = QRadioButton("Point Delete (D)")
        self.select_curve_radio_btn = QRadioButton("Curve Select (C)")

        self.normal_mode_radio_btn.toggled.connect(self.app_data.set_mode_to_normal)

        self.point_pick_mode_radio_btn.toggled.connect(
            self.app_data.set_mode_to_point_picking
        )
        self.point_delete_mode_radio_btn.toggled.connect(
            self.app_data.set_mode_to_point_delete
        )
        self.select_curve_radio_btn.toggled.connect(
            self.app_data.set_mode_to_curve_selection
        )

        self.group_box_layout.addWidget(
            self.normal_mode_radio_btn, alignment=QtCore.Qt.AlignTop
        )
        self.group_box_layout.addWidget(
            self.point_pick_mode_radio_btn, alignment=QtCore.Qt.AlignTop
        )
        self.group_box_layout.addWidget(
            self.point_delete_mode_radio_btn, alignment=QtCore.Qt.AlignTop
        )
        self.group_box_layout.addWidget(
            self.select_curve_radio_btn, alignment=QtCore.Qt.AlignTop
        )

        self.update_radio_btn()
        self.setLayout(self.layout)

    def update_radio_btn(self):
        radio_btn = None
        match self.app_data.mode:
            case AppMode.NORMAL:
                radio_btn = self.normal_mode_radio_btn
            case AppMode.POINT_PICKING:
                radio_btn = self.point_pick_mode_radio_btn
            case AppMode.POINT_DELETE:
                radio_btn = self.point_delete_mode_radio_btn
            case AppMode.CURVE_SELECT:
                radio_btn = self.select_curve_radio_btn

        if not radio_btn.isChecked():
            radio_btn.toggle()

    def delete_mode(self):
        self.app_data.mode = AppMode.POINT_DELETE
        self.update_radio_btn()

    def curve_select_mode(self):
        self.app_data.mode = AppMode.CURVE_SELECT
        self.update_radio_btn()

    def toggle_mode(self):
        match self.app_data.mode:
            case AppMode.NORMAL:
                self.app_data.set_mode_to_point_picking()
            case AppMode.POINT_PICKING:
                self.app_data.set_mode_to_normal()
            case _:
                self.app_data.set_mode_to_point_picking()

        self.update_radio_btn()


class SettingsPanel(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QVBoxLayout(self)
        self.mode_selector = ModeSelector(app_data)
        self.actions_panel = ActionsPanel(app_data)

        self.layout.addWidget(self.mode_selector, alignment=Qt.AlignTop)
        self.layout.addWidget(self.actions_panel, alignment=Qt.AlignTop)
        self.setLayout(self.layout)
        # self.show()


class WindowSizeManager(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.app_data.window_size_manager = self
        self.layout = QHBoxLayout(self)
        self.window_size_lbl = QLabel("Window Size: ")
        self.window_size_inc_btn = QPushButton("+")
        self.window_size_dec_btn = QPushButton("-")
        self.window_size_txt = QLineEdit("")
        self.window_size_txt.setFixedWidth(60)
        self.window_size_dec_btn.setFixedWidth(20)
        self.window_size_inc_btn.setFixedWidth(20)

        self.layout.addWidget(self.window_size_lbl)
        self.layout.addWidget(self.window_size_txt)
        self.layout.addWidget(self.window_size_dec_btn)
        self.layout.addWidget(self.window_size_inc_btn)

        self.window_size_txt.textChanged.connect(self.update_active_curve_window_size)
        self.window_size_dec_btn.clicked.connect(self.decrement_active_window_size)
        self.window_size_inc_btn.clicked.connect(self.increment_active_window_size)

    def set_active_size(self, window_size):
        try:
            text = str(window_size)
            self.window_size_txt.setText(text)
            self.app_data.active_curve.curve.window_size = window_size
            self.app_data.active_curve.update_line()
            self.app_data.main_canvas.draw_idle()

        except ValueError:
            return

    def increment_active_window_size(self):
        current_ws = self.app_data.active_curve.curve.window_size
        # decimal_place = np.log10(current_ws / 10)
        increment = 10 ** np.floor(np.log10(current_ws / 10))
        round_value = np.floor(np.log10(current_ws / 10))
        if round_value > 0:
            round_value = 0
        else:
            round_value = abs(round_value)
        round_value = int(round_value)

        self.set_active_size(float(round(current_ws + increment, round_value)))

    def decrement_active_window_size(self):
        current_ws = self.app_data.active_curve.curve.window_size
        increment = 10 ** np.floor(np.log10(current_ws / 10))
        round_value = np.floor(np.log10(current_ws / 10))
        if round_value > 0:
            round_value = 0
        else:
            round_value = abs(round_value)
        round_value = int(round_value)

        self.set_active_size(float(round(current_ws - increment, round_value)))

    def update_active_curve_window_size(self):
        try:
            self.app_data.active_curve.curve.window_size = float(
                self.window_size_txt.text()
            )
            self.app_data.active_curve.update_line()
            self.app_data.main_canvas.draw_idle()
        except ValueError:
            return

    def update__window_size_text(self):
        try:
            self.app_data.active_curve.curve.window_size = float(
                self.window_size_txt.text()
            )
            self.app_data.active_curve.update_line()
            self.app_data.main_canvas.draw_idle()
        except ValueError:
            return


class ActionsPanel(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        self.save_active_curve_btn = QPushButton("Save Curve (S)")
        self.save_clusters_btn = QPushButton("Save Clusters")
        self.load_clusters_btn = QPushButton("Load Clusters")

        self.save_active_curve_btn.clicked.connect(self.save_curve)
        self.save_clusters_btn.clicked.connect(self.save_clusters)
        self.load_clusters_btn.clicked.connect(self.load_clusters)
        self.layout.addWidget(self.save_active_curve_btn, alignment=QtCore.Qt.AlignTop)
        self.window_size_manager = WindowSizeManager(self.app_data)
        self.layout.addWidget(self.window_size_manager, alignment=QtCore.Qt.AlignTop)
        self.layout.addWidget(self.save_clusters_btn, alignment=QtCore.Qt.AlignTop)
        self.layout.addWidget(self.load_clusters_btn, alignment=QtCore.Qt.AlignTop)

    def load_clusters(self):
        for removed_curve in self.app_data.saved_curves:
            removed_curve.curve.xs = []
            removed_curve.curve.ys = []
            removed_curve.update_line(force_update=True)
        self.app_data.saved_curves = []
        self.app_data.active_curve.curve.clear()
        self.app_data.active_curve.update_line()
        file_name = os.path.join(self.app_data.directory, self.app_data.curve_file)
        curve_df = pl.read_parquet(file_name)
        lbls = curve_df["lbl"].unique()
        self.app_data.interactor_manager.interactors = []
        for lbl in lbls:
            filtered_df = curve_df.filter(pl.col("lbl").eq(lbl))
            xs = [float(val) for val in filtered_df["x"]]
            ys = [float(val) for val in filtered_df["y"]]

            self.app_data.active_curve.curve.xs = xs
            self.app_data.active_curve.curve.ys = ys
            self.app_data.active_curve.curve.window_size = float(
                filtered_df["window_size"][0]
            )

            self.save_curve()

        self.app_data.interactor_manager.add_managed_artist(self.app_data.active_curve)
        self.app_data.main_canvas.draw()

    def save_clusters(self):
        ad = self.app_data
        exported_df = ad.ridges_df.clone()
        self.save_curve()

        if exported_df is None:
            return

        ridges_xs = exported_df["x"]
        ridges_ys = exported_df["y"]

        for idx, ci in enumerate(ad.saved_curves):
            point_in_curve = []
            curve = ci.curve
            for x, y in zip(ridges_xs, ridges_ys):
                if curve.point_is_in_curve(x, y):
                    point_in_curve.append(True)
                else:
                    point_in_curve.append(False)
            exported_df = exported_df.with_columns(
                pl.Series(f"curve_{idx}_contains_pnt", point_in_curve)
            )
        file_name = os.path.join(self.app_data.directory, self.app_data.out_file)
        exported_df.write_parquet(file_name)
        print(f"file {file_name} written")

        curve_df = pl.DataFrame(
            data=[
                pl.Series("x", [], dtype=pl.Float64),
                pl.Series("y", [], dtype=pl.Float64),
                pl.Series("window_size", [], dtype=pl.Float64),
                pl.Series("lbl", [], dtype=pl.Int32),
            ]
        )

        for idx, curve in enumerate(self.app_data.saved_curves):
            print(idx)
            curve = curve.curve
            this_curve_df = pl.DataFrame(data={"x": curve.xs, "y": curve.ys})

            this_curve_df = this_curve_df.with_columns(
                window_size=pl.lit(float(curve.window_size))
            )
            this_curve_df = this_curve_df.with_columns(lbl=pl.lit(int(idx)))
            curve_df = curve_df.vstack(this_curve_df)

        file_name = os.path.join(self.app_data.directory, self.app_data.curve_file)
        curve_df.write_parquet(file_name)
        print(f"file {file_name} written")

    def save_curve(self):
        app_data = self.app_data
        if app_data.active_curve.curve.len() > 1:
            active_curve = app_data.active_curve
            saved_curve = CurveInteractor(
                app_data.main_ax,
                copy.deepcopy(active_curve.curve.xs),
                copy.deepcopy(active_curve.curve.ys),
                app_data,
                window_size=active_curve.curve.window_size,
                status=CurveStatus.INACTIVE,
            )
            saved_curve.update_line(force_update=True)
            app_data.saved_curves.append(saved_curve)
            app_data.interactor_manager.add_managed_artist(saved_curve)
            app_data.active_curve.clear()

        app_data.active_curve.update_line()
        app_data.main_canvas.draw_idle()


class CanvasPanel(QWidget):
    def __init__(self, app_data):
        super().__init__()
        self.app_data = app_data
        self.layout = QVBoxLayout(self)

        main_plot = PlotCanvas(self.app_data, self, width=8, height=6)

        self.layout.addWidget(main_plot)
        self.app_data.nav_bar = NavigationToolbar(main_plot.canvas, self)
        self.layout.addWidget(self.app_data.nav_bar)

        self.setLayout(self.layout)


class PointPickingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        if len(sys.argv) > 1:
            self.arg = sys.argv[1]
        else:
            self.arg = "0"
        self.left = 10
        self.top = 10
        self.title = "Point Picking GUI"
        self.width = 800
        self.height = 900
        self.app_data = AppData(self)

        self.settings_panel = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        wid = QWidget()
        self.setCentralWidget(wid)
        main_layout = QHBoxLayout(wid)
        self.settings_panel = SettingsPanel(app_data=self.app_data)
        self.canvas_panel = CanvasPanel(app_data=self.app_data)

        save_shortcut = QKeySequence(Qt.CTRL + Qt.Key_S)
        save_shortcut = QKeySequence(Qt.Key_S)
        self.save_shortcut = QShortcut(save_shortcut, self)
        self.save_shortcut.activated.connect(
            self.settings_panel.actions_panel.save_curve
        )

        save_shortcut = QKeySequence(Qt.Key_C)
        self.save_shortcut = QShortcut(save_shortcut, self)
        self.save_shortcut.activated.connect(
            self.settings_panel.mode_selector.curve_select_mode
        )
        toggle_shortcut = QKeySequence(Qt.Key_T)
        self.toggle_shortcut = QShortcut(toggle_shortcut, self)
        self.toggle_shortcut.activated.connect(
            self.settings_panel.mode_selector.toggle_mode
        )
        self.settings_panel.setFixedWidth(250)
        delete_shortcut = QKeySequence(Qt.Key_D)
        self.delete_shortcut = QShortcut(delete_shortcut, self)
        self.delete_shortcut.activated.connect(
            self.settings_panel.mode_selector.delete_mode
        )

        main_layout.addWidget(self.settings_panel, alignment=Qt.AlignTop)
        main_layout.addWidget(self.canvas_panel)

        self.app_data.set_mode_to_normal()
        self.app_data.window_size_manager.set_active_size(2)
        self.show()

    @pyqtSlot()
    def fileQuit(self):
        """Quitting the program"""
        self.close()


class PlotCanvas(FigureCanvas):
    def __init__(self, app_data, parent=None, width=8, height=6, dpi=100, arg=0):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.arg = arg
        self.ax = self.fig.add_subplot(111)
        self.app_data = app_data
        FigureCanvas.__init__(self, self.fig)
        self.canvas = self.fig.canvas
        self.setParent(parent)
        self.fig.patch.set_linewidth(3)
        self.fig.patch.set_edgecolor("white")

        FigureCanvas.setSizePolicy(
            self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        raw_df_file = os.path.join(self.app_data.directory, self.app_data.raw_file)
        raw_df = (
            pl.scan_parquet(raw_df_file)
            .select([self.app_data.x_col, self.app_data.y_col])
            .collect()
            .to_pandas()
        )

        ridges_file = os.path.join(self.app_data.directory, self.app_data.in_file)
        ridges_df = pl.read_parquet(ridges_file)
        self.app_data.ridges_df = ridges_df

        self.fig.set_tight_layout(True)

        dsshow(
            raw_df,
            ds.Point(self.app_data.x_col, self.app_data.y_col),
            norm="log",
            aspect="auto",
            ax=self.ax,
            # cmap="Greys_r",
            cmap=cmr.ocean_r,
            # cmap="inferno",
            alpha=0.5,
        )

        min_x, max_x = self.ax.get_xlim()
        min_y, max_y = self.ax.get_ylim()
        x_range = max_x - min_x
        y_range = max_y - min_y
        self.ax.set_xlim(min_x - 0.1 * x_range, max_x + 0.1 * x_range)
        self.ax.set_ylim(min_y - 0.1 * y_range, max_y + 0.1 * y_range)

        self.ax.plot(ridges_df["x"], ridges_df["y"], ".w", ms=0.75)
        self.ax.set_ylabel(self.app_data.y_col)
        self.ax.set_xlabel(self.app_data.x_col)

        self.app_data.active_curve = CurveInteractor(
            self.ax, [], [], self.app_data, status=CurveStatus.ACTIVE
        )
        self.app_data.main_ax = self.ax
        self.app_data.main_canvas = self.canvas
        self.app_data.canvas = self.canvas

        interactors = [self.app_data.active_curve]

        # self is used to have a hard reference, otherwise the
        # manager is garbaged collected and does not work
        self.app_data.interactor_manager = InteractorManager(
            self.fig, self.ax, interactors
        )

        # for curve selection only
        self.canvas.mpl_connect("motion_notify_event", self.set_active_curve_by_closest)
        self.canvas.mpl_connect("button_press_event", self.button_press_callback)

        # self.ax.set_xlim((-2, 2))
        # self.ax.set_ylim((-2, 2))
        # self.ax.grid(True)
        self.draw()

    def set_active_curve_by_closest(self, event):
        if self.app_data.mode != AppMode.CURVE_SELECT:
            return
        if event.inaxes is None:
            return

        x, y = event.xdata, event.ydata
        min_idx = None
        min_dist_sqr = 0
        for idx, curve in enumerate(self.app_data.saved_curves):
            if curve.curve.len() == 0:
                return
            dist_sqr = curve.get_dist_sqr_to_closest(x, y)
            if min_idx is None or dist_sqr < min_dist_sqr:
                min_idx = idx
                min_dist_sqr = dist_sqr

        if min_idx is None:
            return
        active_dist_sqr = self.app_data.active_curve.get_dist_sqr_to_closest(x, y)
        if active_dist_sqr is not None and self.app_data.active_curve.curve.len() >= 2:
            if active_dist_sqr > min_dist_sqr:
                new_saved = copy.deepcopy(self.app_data.active_curve.curve)
                new_active = copy.deepcopy(self.app_data.saved_curves[min_idx].curve)

                self.app_data.active_curve.curve.xs = new_active.xs
                self.app_data.active_curve.curve.ys = new_active.ys
                # self.app_data.active_curve.curve.window_size = new_active.window_size
                self.app_data.window_size_manager.set_active_size(
                    new_active.window_size
                )

                self.app_data.saved_curves[min_idx].curve.xs = new_saved.xs
                self.app_data.saved_curves[min_idx].curve.ys = new_saved.ys
                self.app_data.saved_curves[
                    min_idx
                ].curve.window_size = new_saved.window_size

                self.app_data.active_curve.set_status(CurveStatus.ACTIVE)
                self.app_data.saved_curves[min_idx].set_status(CurveStatus.INACTIVE)

                self.app_data.active_curve.update_line(force_update=True)
                self.app_data.saved_curves[min_idx].update_line(force_update=True)

                self.app_data.interactor_manager.interactors = [
                    curve for curve in self.app_data.saved_curves
                ]
                self.app_data.interactor_manager.add_managed_artist(
                    self.app_data.active_curve
                )

        else:
            new_active = copy.deepcopy(self.app_data.saved_curves[min_idx].curve)
            self.app_data.active_curve.curve.xs = new_active.xs
            self.app_data.active_curve.curve.ys = new_active.ys
            self.app_data.window_size_manager.set_active_size(new_active.window_size)
            self.app_data.active_curve.update_line(force_update=True)

            removed_curve = self.app_data.saved_curves.pop(min_idx)
            removed_curve.curve.xs = []
            removed_curve.curve.ys = []
            removed_curve.update_line(force_update=True)

            self.app_data.active_curve.set_status(CurveStatus.ACTIVE)
            self.app_data.interactor_manager.interactors = [
                curve for curve in self.app_data.saved_curves
            ]
            self.app_data.interactor_manager.add_managed_artist(
                self.app_data.active_curve
            )
        self.app_data.window_size_manager.update_active_curve_window_size()

        self.canvas.draw_idle()

    def button_press_callback(self, event):
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        if self.app_data.mode == AppMode.CURVE_SELECT:
            self.app_data.set_mode_to_normal()
        self.app_data.gui.settings_panel.mode_selector.update_radio_btn()
        self.app_data.window_size_manager.set_active_size(
            self.app_data.active_curve.curve.window_size
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    ex = PointPickingGUI()
    sys.exit(app.exec_())
