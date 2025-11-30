from pyNastran.bdf.bdf import read_bdf
from pyNastran.op2.op2 import read_op2
from pyNastran.bdf.mesh_utils.mass_properties import mass_properties
import pandas as pd
import os, subprocess, psutil
import time
import numpy as np
from skopt import gp_minimize, gbrt_minimize
from scipy.optimize import differential_evolution
from skopt.space import Real
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                                QTextEdit, QProgressBar, QRadioButton, QCheckBox,
                                QComboBox, QFileDialog, QMessageBox, QGroupBox,
                                QFrame, QSplitter, QButtonGroup)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pyvista as pv
from pyvistaqt import QtInteractor
os.environ['QT_API'] = 'pyside6'

class NastranOptimizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CLONE1600")
        
        # Set main window background to match left panel
        self.setStyleSheet("""
            QMainWindow {
                background-color: #e8eaed;
            }
            QMainWindow::separator {
                background-color: #d0d0d0;
                width: 1px;
                height: 1px;
            }
        """)

        self.is_running = False
        self.iteration_data = []
        self.best_result_value = None
        self.best_bdf_name = None
        self.initial_mass = None
        self.load_case = None
        
        self.node_labels_visible = False
        self.element_labels_visible = False
        self.node_label_actors = []
        self.element_label_actors = []

        # Create menu bar FIRST
        self.create_menu_bar()

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left and right panels
        left_panel = self.create_left_panel()
        right_panel = self.create_right_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

    def create_menu_bar(self):
        """Create menu bar with File, Options, and About menus"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #e8eaed;
                color: #202124;
                padding: 2px;
                border-bottom: 1px solid #d0d0d0;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 16px;
                color: #202124;
            }
            QMenuBar::item:selected {
                background-color: #d0d0d0;
            }
            QMenuBar::item:pressed {
                background-color: #c0c0c0;
            }
            QMenu {
                background-color: #ffffff;
                color: #202124;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
            }
            QMenu::item {
                padding: 8px 30px 8px 10px;
                text-align: left;
            }
            QMenu::item:selected {
                background-color: #f0f0f0;
            }
            QMenu::separator {
                height: 1px;
                background: #d0d0d0;
                margin: 4px 8px;
            }
            QMenu::indicator {
                width: 18px;
                height: 18px;
                margin-left: 6px;
            }
            QMenu::indicator:non-exclusive:checked {
                image: none;
                background-color: #1e88e5;
                border: 1px solid #1976d2;
                border-radius: 3px;
            }
            QMenu::indicator:non-exclusive:checked:after {
                content: "✓";
                color: white;
            }
        """)
        
        # FILE MENU
        file_menu = menubar.addMenu("File")
        
        open_bdf_action = file_menu.addAction("📁 Open BDF File...")
        open_bdf_action.setShortcut("Ctrl+O")
        open_bdf_action.triggered.connect(self.browse_file)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction("❌ Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # OPTIONS MENU
        options_menu = menubar.addMenu("Options")
        
        # Add checkable actions for labels
        self.node_labels_action = options_menu.addAction("🔤 Show Node IDs")
        self.node_labels_action.setCheckable(True)
        self.node_labels_action.triggered.connect(self.toggle_node_labels)
        
        self.element_labels_action = options_menu.addAction("🔢 Show Element IDs")
        self.element_labels_action.setCheckable(True)
        self.element_labels_action.triggered.connect(self.toggle_element_labels)
        
        options_menu.addSeparator()
        
        refresh_action = options_menu.addAction("🔄 Refresh Visualization")
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_visual)
        
        # ABOUT MENU
        about_menu = menubar.addMenu("About")
        
        about_action = about_menu.addAction("ℹ️ About")
        about_action.triggered.connect(self.show_about)
        
        help_action = about_menu.addAction("❓ Help")
        help_action.triggered.connect(self.show_help)


    def show_about(self):
        """Show about dialog"""
        about_dialog = QMessageBox(self)
        about_dialog.setWindowTitle("About Nastran Optimizer")
        about_dialog.setTextFormat(Qt.RichText)
        about_dialog.setText(
            "<h3 style='text-align: left;'>Nastran Optimization Tool</h3>"
            "<p style='text-align: left;'>Enhanced with Mass Penalty</p>"
            "<p style='text-align: left;'>Version 1.0</p>"
            "<p style='text-align: left;'>A powerful tool for optimizing Nastran FEM models using advanced optimization algorithms.</p>"
            "<p style='text-align: left;'><b>Features:</b></p>"
            "<ul style='text-align: left;'>"
            "<li>Multiple optimization methods (GP, GBRT, DE)</li>"
            "<li>Real-time 3D visualization</li>"
            "<li>Mass penalty optimization</li>"
            "<li>Interactive result tracking</li>"
            "</ul>"
        )
        about_dialog.setStyleSheet("""
            QMessageBox {
                background-color: #f5f5f5;
            }
            QMessageBox QLabel {
                text-align: left;
                color: #202124;
            }
        """)
        about_dialog.exec()

    def show_help(self):
        """Show help dialog"""
        help_dialog = QMessageBox(self)
        help_dialog.setWindowTitle("Help")
        help_dialog.setTextFormat(Qt.RichText)
        help_dialog.setText(
            "<h3 style='text-align: left;'>Quick Start Guide</h3>"
            "<p style='text-align: left;'><b>1. Load BDF File:</b> Use File → Open BDF File or Browse button</p>"
            "<p style='text-align: left;'><b>2. Set Variables:</b> Enter node/element IDs to monitor (comma-separated)</p>"
            "<p style='text-align: left;'><b>3. Configure Target:</b> Choose result type, component, and optimization mode</p>"
            "<p style='text-align: left;'><b>4. Run Optimization:</b> Click 'Start Optimization' button</p>"
            "<p style='text-align: left;'></p>"
            "<p style='text-align: left;'><b>Visualization Controls:</b></p>"
            "<p style='text-align: left;'>• Options → Show Node IDs: Display node labels</p>"
            "<p style='text-align: left;'>• Options → Show Element IDs: Display element labels</p>"
            "<p style='text-align: left;'>• F5: Refresh visualization</p>"
            "<p style='text-align: left;'></p>"
            "<p style='text-align: left;'><b>Tips:</b></p>"
            "<p style='text-align: left;'>• Use 'All' for properties to optimize all available properties</p>"
            "<p style='text-align: left;'>• Enable Mass Penalty to control mass changes during optimization</p>"
            "<p style='text-align: left;'>• Results are automatically saved to RESULTS.xlsx</p>"
        )
        help_dialog.setStyleSheet("""
            QMessageBox {
                background-color: #f5f5f5;
            }
            QMessageBox QLabel {
                text-align: left;
                color: #202124;
            }
        """)
        help_dialog.exec()

    def toggle_node_labels(self):
        """Toggle node ID labels visibility"""
        if not self.plotter:
            if hasattr(self, 'node_labels_action'):
                self.node_labels_action.setChecked(False)
            return
        
        # Sync state from action if it exists
        if hasattr(self, 'node_labels_action'):
            self.node_labels_visible = self.node_labels_action.isChecked()
        else:
            self.node_labels_visible = not self.node_labels_visible
        
        if self.node_labels_visible:
            self.add_node_labels()
        else:
            self.remove_node_labels()

    def toggle_element_labels(self):
        """Toggle element ID labels visibility"""
        if not self.plotter:
            if hasattr(self, 'element_labels_action'):
                self.element_labels_action.setChecked(False)
            return
        
        # Sync state from action if it exists
        if hasattr(self, 'element_labels_action'):
            self.element_labels_visible = self.element_labels_action.isChecked()
        else:
            self.element_labels_visible = not self.element_labels_visible
        
        if self.element_labels_visible:
            self.add_element_labels()
        else:
            self.remove_element_labels()

    def add_node_labels(self):
        """Add ALL node ID labels in ONE batch operation - OPTIMIZED for performance"""
        if not self.plotter or not os.path.exists(self.bdf_path.text()):
            return
        
        try:
            bdf = read_bdf(self.bdf_path.text())
            
            # Remove old labels first
            self.remove_node_labels()
            
            # Collect ALL positions and labels at once
            positions = []
            labels = []
            
            for nid, node in bdf.nodes.items():
                positions.append(node.get_position())
                labels.append(str(nid))
            
            # Single batch call with AGGRESSIVE optimization settings
            if positions:
                actor = self.plotter.add_point_labels(
                    positions, 
                    labels,
                    font_size=9,  # Smaller font = faster
                    point_color='red',
                    point_size=3,  # Smaller points
                    text_color='yellow',
                    show_points=False,
                    always_visible=False,  # KEY: Don't render occluded labels
                    render_points_as_spheres=False,  # Faster rendering
                    pickable=False,  # Don't allow picking - faster
                    tolerance=0.001,  # More aggressive culling
                    shape_opacity=0.8,  # Slight transparency
                    render=False  # Don't render immediately
                )
                self.node_label_actors.append(actor)
                self.plotter.render()  # Single render at the end
                self.log(f"✓ Displayed {len(labels)} node labels")
            
        except Exception as e:
            self.log(f"Error adding node labels: {e}")


    def add_element_labels(self):
        """Add ALL element ID labels in ONE batch operation - OPTIMIZED for performance"""
        if not self.plotter or not os.path.exists(self.bdf_path.text()):
            return
        
        try:
            bdf = read_bdf(self.bdf_path.text())
            
            # Remove old labels first
            self.remove_element_labels()
            
            # Collect ALL centroids and labels at once
            positions = []
            labels = []
            
            for eid, elem in bdf.elements.items():
                try:
                    # Calculate element centroid
                    node_ids = elem.node_ids
                    elem_positions = [bdf.nodes[nid].get_position() for nid in node_ids if nid in bdf.nodes]
                    
                    if elem_positions:
                        centroid = np.mean(elem_positions, axis=0)
                        positions.append(centroid)
                        labels.append(str(eid))
                except:
                    continue
            
            # Single batch call with AGGRESSIVE optimization
            if positions:
                actor = self.plotter.add_point_labels(
                    positions, 
                    labels,
                    font_size=9,  # Smaller font = faster
                    point_color='blue',
                    point_size=3,  # Smaller points
                    text_color='cyan',
                    show_points=False,
                    always_visible=False,  # KEY: Don't render occluded labels
                    render_points_as_spheres=False,  # Faster rendering
                    pickable=False,  # Don't allow picking - faster
                    tolerance=0.001,  # More aggressive culling
                    shape_opacity=0.8,  # Slight transparency
                    render=False  # Don't render immediately
                )
                self.element_label_actors.append(actor)
                self.plotter.render()  # Single render at the end
                self.log(f"✓ Displayed {len(labels)} element labels")
            
        except Exception as e:
            self.log(f"Error adding element labels: {e}")

    def remove_node_labels(self):
        """Remove all node labels from visualization"""
        if not self.plotter:
            return
        
        for actor in self.node_label_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self.node_label_actors.clear()

    def remove_element_labels(self):
        """Remove all element labels from visualization"""
        if not self.plotter:
            return
        
        for actor in self.element_label_actors:
            try:
                self.plotter.remove_actor(actor)
            except:
                pass
        self.element_label_actors.clear()

    def create_left_panel(self):
        left_widget = QWidget()

        left_widget.setStyleSheet("""
        QWidget {
            background-color: #e8eaed;
        }
        QGroupBox {
            background-color: #f5f5f5;
            border: 2px solid #d0d0d0;
            border-radius: 8px;
            margin-top: 8px;
            padding-top: 15px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            background-color: #f5f5f5;
        }
        QLabel {
            background-color: transparent;
        }
        QLineEdit {
            background-color: white;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            padding: 5px;
        }
        QComboBox {
            background-color: white;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            padding: 5px;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
        }
        QRadioButton {
            background-color: transparent;
        }
        QCheckBox {
            background-color: transparent;
        }
        QPushButton {
            background-color: #ffffff;
            border: 1px solid #c0c0c0;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #f0f0f0;
            border: 2px solid #a0a0a0;
        }
        QPushButton:pressed {
            background-color: #e0e0e0;
        }
    """)
        
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # Title
        """
        title = QLabel("Nastran Optimizer")
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #1e88e5; padding: 10px;")
        left_layout.addWidget(title)
        """

        # FILES Section
        files_group = QGroupBox("📁 FILES")
        files_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        files_layout = QVBoxLayout()
        
        # BDF File
        bdf_layout = QHBoxLayout()
        bdf_label = QLabel("BDF File:")
        bdf_label.setMinimumWidth(100)
        bdf_layout.addWidget(bdf_label)
        self.bdf_path = QLineEdit("...")
        bdf_layout.addWidget(self.bdf_path)
        bdf_btn = QPushButton("Browse")
        bdf_btn.setMaximumWidth(80)
        bdf_btn.clicked.connect(self.browse_file)
        bdf_layout.addWidget(bdf_btn)
        files_layout.addLayout(bdf_layout)
        
        # Nastran Path
        nastran_layout = QHBoxLayout()
        nastran_label = QLabel("Nastran Path:")
        nastran_label.setMinimumWidth(100)
        nastran_layout.addWidget(nastran_label)
        self.nastran_path = QLineEdit("...")
        nastran_layout.addWidget(self.nastran_path)
        nastran_btn = QPushButton("Browse")
        nastran_btn.setMaximumWidth(80)
        nastran_btn.clicked.connect(self.browse_nastran)
        nastran_layout.addWidget(nastran_btn)
        files_layout.addLayout(nastran_layout)
        
        files_group.setLayout(files_layout)
        left_layout.addWidget(files_group)
        
        # VARIABLES Section
        vars_group = QGroupBox("🔧 VARIABLES")
        vars_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        vars_layout = QVBoxLayout()
        
        # Properties
        prop_layout = QHBoxLayout()
        prop_label = QLabel("Properties:")
        prop_label.setMinimumWidth(100)
        prop_layout.addWidget(prop_label)
        self.property_selection = QLineEdit("All")
        prop_layout.addWidget(self.property_selection)
        vars_layout.addLayout(prop_layout)
        
        # Variables
        var_layout = QHBoxLayout()
        var_label = QLabel("Responses:")
        var_label.setMinimumWidth(100)
        var_layout.addWidget(var_label)
        self.variables = QLineEdit("2143, 225")
        var_layout.addWidget(self.variables)
        vars_layout.addLayout(var_layout)
        
        # Method
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        method_label.setMinimumWidth(100)
        method_layout.addWidget(method_label)
        self.optimization_method = QComboBox()
        self.optimization_method.addItems(["Gaussian Process", "Boosted Trees", "Differential Evo"])
        self.optimization_method.setMinimumWidth(400)
        method_layout.addWidget(self.optimization_method)
        method_layout.addStretch()  # Push everything to the left
        method_label.setMinimumWidth(100)
        vars_layout.addLayout(method_layout)
        
        # Iterations
        iter_layout = QHBoxLayout()
        iter_label = QLabel("Iterations:")
        iter_label.setMinimumWidth(100)
        iter_layout.addWidget(iter_label)
        self.n_calls = QLineEdit("30")
        iter_layout.addWidget(self.n_calls)
        vars_layout.addLayout(iter_layout)
        
        # Multipliers
        mult_layout = QHBoxLayout()
        mult_label = QLabel("Multipliers:")
        mult_label.setMinimumWidth(100)
        mult_layout.addWidget(mult_label)
        mult_layout.addWidget(QLabel("Min:"))
        self.min_bound = QLineEdit("0.1")
        self.min_bound.setMaximumWidth(45)
        mult_layout.addWidget(self.min_bound)
        mult_layout.addWidget(QLabel("Max:"))
        self.max_bound = QLineEdit("5.0")
        self.max_bound.setMaximumWidth(45)
        mult_layout.addWidget(self.max_bound)
        mult_layout.addStretch()
        vars_layout.addLayout(mult_layout)
        
        vars_group.setLayout(vars_layout)
        left_layout.addWidget(vars_group)
        
        # TARGET Section
        target_group = QGroupBox("🎯 TARGET")
        target_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        target_layout = QVBoxLayout()
        
        # Result Type
        rt_layout = QHBoxLayout()
        rt_label = QLabel("Result Type:")
        rt_label.setMinimumWidth(100)
        rt_layout.addWidget(rt_label)
        self.result_type_group = QButtonGroup()
        self.rb_displacement = QRadioButton("Displacement")
        self.rb_cbush = QRadioButton("CBUSH Force")
        self.rb_displacement.setChecked(True)
        self.result_type_group.addButton(self.rb_displacement)
        self.result_type_group.addButton(self.rb_cbush)
        self.rb_displacement.toggled.connect(self.create_component_options)
        rt_layout.addWidget(self.rb_displacement)
        rt_layout.addWidget(self.rb_cbush)
        rt_layout.addStretch()
        target_layout.addLayout(rt_layout)
        
        # Component
        comp_layout = QHBoxLayout()
        comp_label = QLabel("Component:")
        comp_label.setMinimumWidth(100)
        comp_layout.addWidget(comp_label)
        self.comp_group = QButtonGroup()
        self.comp_buttons = {}
        for comp in ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]:
            rb = QRadioButton(comp)
            self.comp_buttons[comp] = rb
            self.comp_group.addButton(rb)
            comp_layout.addWidget(rb)
        self.comp_buttons["XYZ"].setChecked(True)
        comp_layout.addStretch()
        target_layout.addLayout(comp_layout)
        
        # Objective Function
        obj_layout = QHBoxLayout()
        obj_label = QLabel("Obj Function:")
        obj_label.setMinimumWidth(100)
        obj_layout.addWidget(obj_label)
        self.objective_function = QLineEdit("w1+w2")
        obj_layout.addWidget(self.objective_function)
        target_layout.addLayout(obj_layout)
        
        # Mode
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setMinimumWidth(100)
        mode_layout.addWidget(mode_label)
        self.mode_group = QButtonGroup()
        self.rb_minimize = QRadioButton("Minimize")
        self.rb_maximize = QRadioButton("Maximize")
        self.rb_target = QRadioButton("Target")
        self.rb_minimize.setChecked(True)
        self.mode_group.addButton(self.rb_minimize)
        self.mode_group.addButton(self.rb_maximize)
        self.mode_group.addButton(self.rb_target)
        self.rb_target.toggled.connect(self.update_target_state)
        mode_layout.addWidget(self.rb_minimize)
        mode_layout.addWidget(self.rb_maximize)
        mode_layout.addWidget(self.rb_target)
        mode_layout.addWidget(QLabel("Value:"))
        self.target_value = QLineEdit("1.0")
        self.target_value.setMaximumWidth(45)
        self.target_value.setEnabled(False)
        mode_layout.addWidget(self.target_value)
        mode_layout.addStretch()
        target_layout.addLayout(mode_layout)
        
        # Mass Penalty
        mass_layout = QHBoxLayout()
        mass_label = QLabel("Mass Penalty:")
        mass_label.setMinimumWidth(100)
        mass_layout.addWidget(mass_label)
        self.use_mass_penalty = QCheckBox("Enable")
        self.use_mass_penalty.toggled.connect(self.update_mass_state)
        mass_layout.addWidget(self.use_mass_penalty)
        mass_layout.addWidget(QLabel("Factor:"))
        self.mass_penalty_factor = QLineEdit("1.0")
        self.mass_penalty_factor.setMaximumWidth(45)
        self.mass_penalty_factor.setEnabled(False)
        mass_layout.addWidget(self.mass_penalty_factor)
        mass_layout.addStretch()
        target_layout.addLayout(mass_layout)
        
        target_group.setLayout(target_layout)
        left_layout.addWidget(target_group)
        
        left_layout.addSpacing(10)  # 15 pixels of space
        # Progress
        prog_layout = QHBoxLayout()
        prog_layout.addWidget(QLabel("Progress:"))
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
        QProgressBar {
            border: 2px solid #d0d0d0;
            border-radius: 0px;
            text-align: center;
            background-color: white;
            color: #202124;
            font-weight: bold;
            height: 4px;            
        }
        QProgressBar::chunk {
            background-color: #4caf50;
            border-radius: 0px;
        }
        """)
        prog_layout.addWidget(self.progress)
        left_layout.addLayout(prog_layout)
        
        # Status
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        left_layout.addLayout(status_layout)
        
        # Best Result
        best_layout = QHBoxLayout()
        best_layout.addWidget(QLabel("Best Result:"))
        self.best_result_label = QLabel("N/A")
        self.best_result_label.setFont(QFont('Arial', 12, QFont.Bold))
        self.best_result_label.setStyleSheet("color: #1e88e5;")
        best_layout.addWidget(self.best_result_label)
        best_layout.addStretch()
        left_layout.addLayout(best_layout)
        
        # Mass
        mass_info_layout = QHBoxLayout()
        mass_info_layout.addWidget(QLabel("Current Mass:"))
        self.mass_label = QLabel("N/A")
        self.mass_label.setStyleSheet("color: #1e88e5;")
        mass_info_layout.addWidget(self.mass_label)
        mass_info_layout.addStretch()
        left_layout.addLayout(mass_info_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("⚡ Start Optimization")
        self.start_btn.setStyleSheet("background-color: #4caf50; color: white; font-weight: bold; padding: 8px;")
        self.start_btn.clicked.connect(self.start_optimization)
        
        self.refresh_btn = QPushButton("🔄 Refresh Visual")
        self.refresh_btn.setStyleSheet("background-color: #2196f3; color: white; font-weight: bold; padding: 8px;")
        self.refresh_btn.clicked.connect(self.refresh_visual)
        
        self.stop_btn = QPushButton("⬛ Stop")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 8px;")
        self.stop_btn.clicked.connect(self.stop_optimization)
        self.stop_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)

        left_layout.addSpacing(15)  # 15 pixels of space

        # Log (stretched to bottom)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #f5f5f5; border: 1px solid #ccc;")
        left_layout.addWidget(self.log_text, stretch=1)  # This makes it stretch

        return left_widget

    def create_right_panel(self):
        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: #0d2137;")  # Match PyVista background
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # PyVista viewer (2/3 of space)
        try:
            self.plotter = QtInteractor(right_widget)
            self.plotter.add_axes(color="white", viewport=(0.9, 0.8, 1.1, 1.0), interactive=True)
            self.plotter.set_background('#0d2137')
            
            # Remove all margins, padding, and borders
            self.plotter.setContentsMargins(0, 0, 0, 0)
            self.plotter.interactor.setContentsMargins(0, 0, 0, 0)
            self.plotter.setStyleSheet("border: 0px; margin: 0px; padding: 0px;")
            self.plotter.interactor.setStyleSheet("border: 0px; margin: 0px; padding: 0px; background-color: #0d2137;")
            right_layout.addWidget(self.plotter.interactor, stretch=2)

        except Exception as e:
            placeholder = QLabel("3D Viewer (PyVista not available)")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #0d2137; color: white;")
            right_layout.addWidget(placeholder, stretch=2)
            print(f"PyVista initialization failed: {e}")
            self.plotter = None
        
        # Matplotlib plots on bottom (1/3 of space)
        plot_widget = QWidget()
        plot_widget.setStyleSheet("background-color: #0a1929;")
        plot_layout = QHBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        
        # Create matplotlib figure with aerospace blue theme
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(12, 4), dpi=100, facecolor='#0d2137')

        self.ax1 = self.fig.add_subplot(121)
        self.ax1.set_facecolor('#0d2137')
        self.ax1.set_xlabel('Iteration', color='white')
        self.ax1.set_ylabel('Result Value', color='white')
        self.ax1.set_title('Optimization Progress', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.2, color='#1e88e5')
        
        self.ax2 = self.fig.add_subplot(122)
        self.ax2.set_facecolor('#0d2137')
        #34x495e
        self.ax2.set_xlabel('Iteration', color='white')
        self.ax2.set_ylabel('Best Result', color='white')
        self.ax2.set_title('Best Result Evolution', color='white')
        self.ax2.tick_params(colors='white')
        self.ax2.grid(True, alpha=0.2, color='#1e88e5')
        
        self.fig.tight_layout(pad=1.5)
        
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet("background-color: #0a1929;")
        plot_layout.addWidget(self.canvas)
        
        right_layout.addWidget(plot_widget, stretch=1)
        
        return right_widget
    
    def refresh_visual(self):
        """Refresh the PyVista visualization with current BDF"""
        bdf_file = self.bdf_path.text()
        if os.path.exists(bdf_file):
            self.log("Refreshing visualization...")
            self.update_pyvista_mesh(bdf_file)
            self.log("Visualization refreshed")
        else:
            self.log("Error: BDF file not found")
            QMessageBox.warning(self, "Warning", "Please select a valid BDF file first!")

    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select BDF File", "", "BDF files (*.bdf);;All files (*.*)")
        if filename:
            self.bdf_path.setText(filename)
            # Visualize the BDF immediately after browsing
            self.log(f"Loading BDF file for visualization: {filename}")
            self.update_pyvista_mesh(filename)
            self.log("BDF file loaded and visualized")
        
    def browse_nastran(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Nastran Executable", "", "Executable files (*.exe);;All files (*.*)")
        if filename:
            self.nastran_path.setText(filename)
    
    def update_target_state(self):
        self.target_value.setEnabled(self.rb_target.isChecked())
    
    def update_mass_state(self):
        self.mass_penalty_factor.setEnabled(self.use_mass_penalty.isChecked())
    
    def create_component_options(self):
        # Component options are already created, no need to recreate
        pass
    
    def get_result_type(self):
        return "displacement" if self.rb_displacement.isChecked() else "cbush_force"
    
    def get_displacement_component(self):
        for comp, rb in self.comp_buttons.items():
            if rb.isChecked():
                return comp
        return "XYZ"
    
    def get_optimize_mode(self):
        if self.rb_minimize.isChecked():
            return "minimize"
        elif self.rb_maximize.isChecked():
            return "maximize"
        else:
            return "target"
    
    def parse_property_selection(self, prop_string, all_property_ids):
        prop_string = prop_string.strip().lower()
        if prop_string == "all":
            return all_property_ids
        selected_ids = []
        parts = [p.strip() for p in prop_string.split(',')]
        for part in parts:
            if '-' in part:
                try:
                    start, end = part.split('-')
                    start_id, end_id = int(start.strip()), int(end.strip())
                    for pid in all_property_ids:
                        if start_id <= pid <= end_id:
                            selected_ids.append(pid)
                except ValueError:
                    raise ValueError(f"Invalid range format: {part}")
            else:
                try:
                    pid = int(part)
                    if pid in all_property_ids:
                        selected_ids.append(pid)
                    else:
                        self.log(f"Warning: Property {pid} not found in model")
                except ValueError:
                    raise ValueError(f"Invalid property ID: {part}")
        return list(set(selected_ids))
    
    def get_mass(self, bdf):
        try:
            mass, cg, I = mass_properties(bdf)
            return mass
        except Exception as e:
            self.log(f"Warning: Could not calculate mass: {e}")
            return None
    
    def get_displacement_value(self, node_id, df, component):
        values = df.loc[df["NodeID"] == node_id, component]
        if values.empty:
            raise ValueError(f"Node {node_id} not found")
        return values.iat[0]
    
    def get_cbush_force_value(self, element_id, df, component):
        values = df.loc[df["ElementID"] == element_id, component]
        if values.empty:
            raise ValueError(f"Element {element_id} not found")
        return values.iat[0]
    
    def calculate_tmag(self, node_id, df, component):
        if component == "XYZ":
            t1 = self.get_displacement_value(node_id, df, "t1")
            t2 = self.get_displacement_value(node_id, df, "t2")
            t3 = self.get_displacement_value(node_id, df, "t3")
            return np.sqrt(t1**2 + t2**2 + t3**2)
        elif component == "XY":
            t1 = self.get_displacement_value(node_id, df, "t1")
            t2 = self.get_displacement_value(node_id, df, "t2")
            return np.sqrt(t1**2 + t2**2)
        elif component == "XZ":
            t1 = self.get_displacement_value(node_id, df, "t1")
            t3 = self.get_displacement_value(node_id, df, "t3")
            return np.sqrt(t1**2 + t3**2)
        elif component == "YZ":
            t2 = self.get_displacement_value(node_id, df, "t2")
            t3 = self.get_displacement_value(node_id, df, "t3")
            return np.sqrt(t2**2 + t3**2)
        else:
            comp_map = {"X": "t1", "Y": "t2", "Z": "t3"}
            return self.get_displacement_value(node_id, df, comp_map.get(component, "t1"))
    
    def calculate_fmag(self, element_id, df, component):
        if component == "XYZ":
            fx = self.get_cbush_force_value(element_id, df, "fx")
            fy = self.get_cbush_force_value(element_id, df, "fy")
            fz = self.get_cbush_force_value(element_id, df, "fz")
            return np.sqrt(fx**2 + fy**2 + fz**2)
        elif component == "XY":
            fx = self.get_cbush_force_value(element_id, df, "fx")
            fy = self.get_cbush_force_value(element_id, df, "fy")
            return np.sqrt(fx**2 + fy**2)
        elif component == "XZ":
            fx = self.get_cbush_force_value(element_id, df, "fx")
            fz = self.get_cbush_force_value(element_id, df, "fz")
            return np.sqrt(fx**2 + fz**2)
        elif component == "YZ":
            fy = self.get_cbush_force_value(element_id, df, "fy")
            fz = self.get_cbush_force_value(element_id, df, "fz")
            return np.sqrt(fy**2 + fz**2)
        else:
            comp_map = {"X": "fx", "Y": "fy", "Z": "fz"}
            return self.get_cbush_force_value(element_id, df, comp_map.get(component, "fx"))
    
    def evaluate_objective_function(self, node_values, func_str):
        namespace = {'abs': abs, 'sqrt': np.sqrt, 'np': np, 'sin': np.sin, 'cos': np.cos, 
                    'tan': np.tan, 'exp': np.exp, 'log': np.log, '__builtins__': {}}
        namespace.update(node_values)
        try:
            result = eval(func_str, namespace)
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluating objective function: {e}")
    
    def apply_mass_penalty(self, result, current_mass, mode):
        if not self.use_mass_penalty.isChecked() or self.initial_mass is None or current_mass is None:
            return result
        mass_change = (current_mass - self.initial_mass) / self.initial_mass
        penalty_factor = float(self.mass_penalty_factor.text())
        if mode == 'minimize':
            penalized = result * (1.0 + penalty_factor * max(0, mass_change))
        elif mode == 'maximize':
            penalized = result / (1.0 + penalty_factor * max(0, mass_change))
        elif mode == 'target':
            penalized = result * (1.0 + penalty_factor * max(0, mass_change))
        else:
            penalized = result
        return penalized
    
    def extract_results_from_op2(self, op2_name, variables, result_type):
        """Extract variable values from OP2 file - unified function"""
        try:
            op2 = read_op2(op2_name, build_dataframe=True)
            if self.load_case is None:
                self.load_case = list(op2.displacements.keys())[0]
            
            if result_type == "displacement":
                df = op2.displacements[self.load_case].dataframe
                df.reset_index(inplace=True)
            else:
                df = op2.cbush_force[self.load_case].dataframe
                df.reset_index(inplace=True)
            
            comp = self.get_displacement_component()
            variable_values = {}
            
            for i, id_value in enumerate(variables, 1):
                if result_type == "displacement":
                    if comp in ["XYZ", "XY", "XZ", "YZ"]:
                        value = self.calculate_tmag(id_value, df, comp)
                    else:
                        value = self.get_displacement_value(id_value, df, comp)
                else:
                    if comp in ["XYZ", "XY", "XZ", "YZ"]:
                        value = self.calculate_fmag(id_value, df, comp)
                    else:
                        value = self.get_cbush_force_value(id_value, df, comp)
                variable_values[f'w{i}'] = value
            return variable_values
        except Exception as e:
            self.log(f"Error extracting results from {op2_name}: {e}")
            return None
    
    def log(self, message):
        self.log_text.append(f"{time.strftime('%H:%M:%S')} - {message}")
        QApplication.processEvents()
    
    @Slot(str)
    def update_pyvista_mesh(self, bdf_path):   
        """Update PyVista visualization with current BDF mesh - OPTIMIZED"""
        if self.plotter is None:
            self.log("PyVista viewer not available")
            return
            
        if not os.path.exists(bdf_path):
            self.log(f"BDF file not found: {bdf_path}")
            return
        
        # Clear existing labels
        self.remove_node_labels()
        self.remove_element_labels()
        self.node_labels_visible = False
        self.element_labels_visible = False
        if hasattr(self, 'node_labels_action'):
            self.node_labels_action.setChecked(False)
        if hasattr(self, 'element_labels_action'):
            self.element_labels_action.setChecked(False)
        
        try:
            bdf = read_bdf(bdf_path)

            # Extract nodes and create mapping
            nodes = []
            node_ids = []
            node_id_to_idx = {}
            for nid, node in bdf.nodes.items():
                node_id_to_idx[nid] = len(nodes)
                nodes.append(node.get_position())
                node_ids.append(nid)
            
            if not nodes:
                self.log("No nodes found in BDF")
                return
            
            points = np.array(nodes)
            
            # Get selected variables (nodes or elements to monitor)
            variables_str = self.variables.text().strip()
            selected_variables = [int(x.strip()) for x in variables_str.split(',') if x.strip()] if variables_str else []
            selected_variables_set = set(selected_variables)
            result_type = self.get_result_type()
            
            # Get selected properties
            all_property_ids = list(bdf.properties.keys())
            try:
                selected_property_ids = self.parse_property_selection(
                    self.property_selection.text(), all_property_ids
                )
                selected_property_ids_set = set(selected_property_ids)
                all_properties_selected = (len(selected_property_ids) == len(all_property_ids))
            except:
                selected_property_ids_set = set()
                all_properties_selected = True
            
            self.plotter.clear()
            
            # ==================== SHELL ELEMENTS (BATCH PROCESSING) ====================
            selected_shell_cells = []
            unselected_shell_cells = []
            
            for eid, elem in bdf.elements.items():
                if elem.type in ['CQUAD4', 'CQUAD8', 'CQUAD']:
                    nids = elem.node_ids[:4]
                    if all(nid in node_id_to_idx for nid in nids):
                        indices = [node_id_to_idx[nid] for nid in nids]
                        cell = [4] + indices
                        
                        elem_pid = elem.pid if hasattr(elem, 'pid') else None
                        if all_properties_selected or elem_pid in selected_property_ids_set:
                            selected_shell_cells.append(cell)
                        else:
                            unselected_shell_cells.append(cell)
                            
                elif elem.type in ['CTRIA3', 'CTRIA6', 'CTRIA']:
                    nids = elem.node_ids[:3]
                    if all(nid in node_id_to_idx for nid in nids):
                        indices = [node_id_to_idx[nid] for nid in nids]
                        cell = [3] + indices
                        
                        elem_pid = elem.pid if hasattr(elem, 'pid') else None
                        if all_properties_selected or elem_pid in selected_property_ids_set:
                            selected_shell_cells.append(cell)
                        else:
                            unselected_shell_cells.append(cell)
            
            # Plot unselected shells (gray)
            if unselected_shell_cells:
                unsel_array = np.hstack(unselected_shell_cells)
                unsel_mesh = pv.PolyData(points, unsel_array)
                self.plotter.add_mesh(unsel_mesh, color='gray', show_edges=True, opacity=0.3)
            
            # Plot selected shells (cyan)
            if selected_shell_cells:
                sel_array = np.hstack(selected_shell_cells)
                sel_mesh = pv.PolyData(points, sel_array)
                self.plotter.add_mesh(sel_mesh, color='cyan', show_edges=True, opacity=0.8)
                self.log(f"Mesh: {len(nodes)} nodes, {len(selected_shell_cells)} selected shells, {len(unselected_shell_cells)} unselected shells")
            
            # ==================== CBUSH ELEMENTS (BATCH PROCESSING WITH LINES) ====================
            cbush_points_selected = []
            cbush_lines_selected = []
            cbush_points_unselected = []
            cbush_lines_unselected = []
            
            for eid, elem in bdf.elements.items():
                if elem.type == 'CBUSH':
                    try:
                        nids = elem.node_ids[:2]
                        if all(nid in node_id_to_idx for nid in nids):
                            pos1 = bdf.nodes[nids[0]].get_position()
                            pos2 = bdf.nodes[nids[1]].get_position()
                            
                            if result_type == 'cbush_force' and eid in selected_variables_set:
                                idx = len(cbush_points_selected)
                                cbush_points_selected.extend([pos1, pos2])
                                cbush_lines_selected.append([2, idx, idx+1])
                            else:
                                idx = len(cbush_points_unselected)
                                cbush_points_unselected.extend([pos1, pos2])
                                cbush_lines_unselected.append([2, idx, idx+1])
                    except (KeyError, IndexError, AttributeError):
                        continue
            
            # Plot unselected CBUSH (yellow)
            if cbush_lines_unselected:
                cbush_unsel_points = np.array(cbush_points_unselected)
                cbush_unsel_lines = np.hstack(cbush_lines_unselected)
                cbush_unsel_mesh = pv.PolyData(cbush_unsel_points, lines=cbush_unsel_lines)
                self.plotter.add_mesh(cbush_unsel_mesh, color='yellow', line_width=6)
            
            # Plot selected CBUSH (red)
            if cbush_lines_selected:
                cbush_sel_points = np.array(cbush_points_selected)
                cbush_sel_lines = np.hstack(cbush_lines_selected)
                cbush_sel_mesh = pv.PolyData(cbush_sel_points, lines=cbush_sel_lines)
                self.plotter.add_mesh(cbush_sel_mesh, color='red', line_width=12)
            
            cbush_total = len(cbush_lines_selected) + len(cbush_lines_unselected)
            if cbush_total > 0:
                self.log(f"CBUSH: {cbush_total} elements ({len(cbush_lines_selected)} highlighted)")
            
            # ==================== CBAR ELEMENTS (BATCH PROCESSING WITH LINES) ====================
            cbar_points = []
            cbar_lines = []
            
            for eid, elem in bdf.elements.items():
                if elem.type == 'CBAR':
                    try:
                        nids = elem.node_ids[:2]
                        if all(nid in node_id_to_idx for nid in nids):
                            pos1 = bdf.nodes[nids[0]].get_position()
                            pos2 = bdf.nodes[nids[1]].get_position()
                            idx = len(cbar_points)
                            cbar_points.extend([pos1, pos2])
                            cbar_lines.append([2, idx, idx+1])
                    except (KeyError, IndexError, AttributeError):
                        continue
            
            # Plot all CBAR (green)
            if cbar_lines:
                cbar_points_array = np.array(cbar_points)
                cbar_lines_array = np.hstack(cbar_lines)
                cbar_mesh = pv.PolyData(cbar_points_array, lines=cbar_lines_array)
                self.plotter.add_mesh(cbar_mesh, color='green', line_width=5)
                self.log(f"CBAR: {len(cbar_lines)} elements")
            
            # ==================== HIGHLIGHTED NODES (for displacement monitoring) ====================
            if result_type == 'displacement' and selected_variables:
                monitored_points = []
                for nid in selected_variables:
                    if nid in node_id_to_idx:
                        idx = node_id_to_idx[nid]
                        monitored_points.append(points[idx])
                
                if monitored_points:
                    monitored_points = np.array(monitored_points)
                    x_range = np.ptp(points[:, 0])
                    y_range = np.ptp(points[:, 1])
                    z_range = np.ptp(points[:, 2])
                    avg_range = (x_range + y_range + z_range) / 3
                    sphere_radius = avg_range * 0.015  # 1.5% of average dimension
                    
                    sphere_cloud = pv.PolyData(monitored_points)
                    spheres = sphere_cloud.glyph(geom=pv.Sphere(radius=sphere_radius), scale=False)
                    self.plotter.add_mesh(spheres, color='red', opacity=1.0)
                    
                    self.log(f"Highlighted {len(monitored_points)} monitored nodes")
            
            self.plotter.add_axes(color="white",viewport=(0.9, 0.8, 1.1, 1.0),interactive=True)
            self.plotter.reset_camera()
            
        except Exception as e:
            import traceback
            self.log(f"Could not update PyVista mesh: {e}")
            self.log(traceback.format_exc())

    def update_plots(self):
        if not self.iteration_data:
            return
        iterations = [d['iteration'] for d in self.iteration_data]
        results = [d['result'] for d in self.iteration_data]
        best_results = [d['best_so_far'] for d in self.iteration_data]
        
        self.ax1.clear()
        self.ax1.plot(iterations, results, color='#42a5f5', marker='o', label='Current Result', markersize=4, linewidth=2)
        self.ax1.set_xlabel('Iteration', color='white')
        self.ax1.set_ylabel('Result Value', color='white')
        self.ax1.set_title('Optimization Progress', color='white')
        self.ax1.set_facecolor('#0d2137')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(True, alpha=0.2, color='#1e88e5')
        self.ax1.legend(facecolor='#0d2137', edgecolor='#1e88e5', labelcolor='white')
        if self.get_optimize_mode() == 'target':
            self.ax1.axhline(y=float(self.target_value.text()), color='#ff6b6b', linestyle='--', label='Target', linewidth=2)
            self.ax1.legend(facecolor='#0d2137', edgecolor='#1e88e5', labelcolor='white')
        
        self.ax2.clear()
        self.ax2.plot(iterations, best_results, color='#66bb6a', marker='o', label='Best Result', markersize=4, linewidth=2)
        self.ax2.set_xlabel('Iteration', color='white')
        self.ax2.set_ylabel('Best Result', color='white')
        self.ax2.set_title('Best Result Evolution', color='white')
        self.ax2.set_facecolor('#0d2137')
        self.ax2.tick_params(colors='white')
        self.ax2.grid(True, alpha=0.2, color='#1e88e5')
        self.ax2.legend(facecolor='#0d2137', edgecolor='#1e88e5', labelcolor='white')
        if self.get_optimize_mode() == 'target':
            self.ax2.axhline(y=float(self.target_value.text()), color='#ff6b6b', linestyle='--', label='Target', linewidth=2)
            self.ax2.legend(facecolor='#0d2137', edgecolor='#1e88e5', labelcolor='white')
        
        self.canvas.draw()
    
    def start_optimization(self):
        if not os.path.exists(self.bdf_path.text()):
            QMessageBox.critical(self, "Error", "BDF file not found!")
            return
        
        if not os.path.exists(self.nastran_path.text()):
            QMessageBox.critical(self, "Error", "Nastran executable not found!")
            return
    
        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.iteration_data = []
        self.initial_mass = None
        self.load_case = None

        # Load initial mesh in PyVista
        self.update_pyvista_mesh(self.bdf_path.text())
        
        if self.plotter:
            self.plotter.setEnabled(False)  # Disable the widget
            self.overlay_actor = self.plotter.add_text(
            "Visualization is locked during the optimization...",
            position='upper_left',
            font_size=12,
            color='gray',
            font='arial',
            shadow=True
            )
            
        # Start optimization in separate thread
        self.opt_thread = OptimizationThread(self)
        self.opt_thread.progress_signal.connect(self.update_progress)
        self.opt_thread.finished_signal.connect(self.optimization_finished)
        self.opt_thread.log_signal.connect(self.log)
        self.opt_thread.mass_signal.connect(self.mass_label.setText)
        self.opt_thread.mesh_update_signal.connect(self.update_pyvista_mesh)  # ADD THIS
        self.opt_thread.start()
    
    def stop_optimization(self):
        self.is_running = False
        self.log("Optimization stopped by user")
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        

    def update_progress(self, iteration, result, best_result, current_mass, is_new_best):
        progress_pct = (iteration / int(self.n_calls.text())) * 100
        self.progress.setValue(int(progress_pct))
        self.best_result_label.setText(f"{best_result:.5f}")
        
        self.iteration_data.append({
        'iteration': iteration,
        'result': result,
        'best_so_far': best_result,
        'mass': current_mass
        })

        if current_mass is not None and self.initial_mass:
            mass_change_pct = ((current_mass - self.initial_mass) / self.initial_mass * 100)
            self.mass_label.setText(f"{current_mass:.2f} ({mass_change_pct:+.2f}%)")
        
        marker = " ★" if is_new_best else ""
        mass_info = f" | Mass: {current_mass:.5f}" if current_mass is not None else ""
        self.log(f"[{iteration}/{self.n_calls.text()}] Result: {result:.5f} | Best: {best_result:.5f}{mass_info}{marker}")

        self.update_plots()
        QApplication.processEvents()
    
    def optimization_finished(self, success, message):
        if success:
            self.log("Optimization complete!")
            self.status_label.setText("Complete")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            
            finish_dialog = QMessageBox(self)
            finish_dialog.setWindowTitle("Finished")
            finish_dialog.setTextFormat(Qt.RichText)
            finish_dialog.setText(
                "<h2 style='margin:0; color:#202124;'>Optimization is complete!</h2>"
                "<p style='margin-top:8px; color:#5f6368;'>Your Nastran model has been successfully optimized.</p>"
            )
            finish_dialog.setStandardButtons(QMessageBox.Ok)
            finish_dialog.setStyleSheet("""
                QMessageBox {
                    background-color: #e8eaed;
                    color: #202124;
                    border: 1px solid #d0d0d0;
                    border-radius: 6px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 12pt;
                }
                QMessageBox QLabel {
                    color: #202124;
                }
                QMessageBox QPushButton {
                    background-color: #ffffff;
                    color: #202124;
                    padding: 6px 12px;
                    border: 1px solid #d0d0d0;
                    border-radius: 4px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #f0f0f0;
                }
                QMessageBox QPushButton:pressed {
                    background-color: #d0d0d0;
                }
            """)
            finish_dialog.exec()

        else:
            self.log(f"ERROR: {message}")
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Optimization Error", message)
        
        # RE-ENABLE interaction after optimization
        if self.plotter:
            self.plotter.setEnabled(True)  # Re-enable the widget
            self.plotter.remove_actor(self.overlay_actor)
            self.overlay_actor = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.is_running = False
        
        
    def save_results(self, property_ids, original_values, best_multipliers, history, best_result, best_mass):
        results_data = []
        for i, pid in enumerate(property_ids):
            if original_values[pid] is None:
                continue
            attr_name, original = original_values[pid]
            multiplier = best_multipliers[i]
            results_data.append({
                'PID': pid, 
                'Property_Type': attr_name, 
                'Multiplier': multiplier, 
                'Original_Value': original, 
                'New_Value': original * multiplier
            })
        
        df_results = pd.DataFrame(results_data)
        df_history = pd.DataFrame(history)
        
        summary_data = {
            'Parameter': [
                'Best Result Value', 'Best BDF File', 'Optimization Mode', 'Target Value', 
                'Total Iterations', 'Result Type', 'Component', 'Load Case', 
                'Objective Function', 'Properties Optimized', 'Property Selection', 
                'Mass Penalty Enabled', 'Mass Penalty Factor', 'Initial Mass', 
                'Best Solution Mass', 'Mass Change (%)'
            ],
            'Value': [
                f"{best_result:.6f}", 
                self.best_bdf_name if self.best_bdf_name else "N/A", 
                self.get_optimize_mode(), 
                f"{self.target_value.text()}" if self.get_optimize_mode() == 'target' else "N/A", 
                len(history), 
                self.get_result_type(), 
                self.get_displacement_component().upper(), 
                self.load_case, 
                self.objective_function.text(), 
                len(property_ids), 
                self.property_selection.text(), 
                "Yes" if self.use_mass_penalty.isChecked() else "No", 
                f"{self.mass_penalty_factor.text()}" if self.use_mass_penalty.isChecked() else "N/A", 
                f"{self.initial_mass:.2f}" if self.initial_mass else "N/A", 
                f"{best_mass:.2f}" if best_mass else "N/A", 
                f"{((best_mass - self.initial_mass) / self.initial_mass * 100):+.2f}%" if (self.initial_mass and best_mass) else "N/A"
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        
        with pd.ExcelWriter("RESULTS.xlsx", engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_results.to_excel(writer, sheet_name='Best_Solution', index=False)
            df_history.to_excel(writer, sheet_name='History', index=False)
        self.log("Results saved to RESULTS.xlsx")


class OptimizationThread(QThread):
    progress_signal = Signal(int, float, float, object, bool)
    finished_signal = Signal(bool, str)
    log_signal = Signal(str)  # ADD THIS LINE
    mass_signal = Signal(str)  # ADD THIS
    mesh_update_signal = Signal(str)

    def __init__(self, gui):
        super().__init__()
        self.gui = gui
    
    def wait_for_nastran(self):
        while True:
            running = [p for p in psutil.process_iter(["name"]) if p.info["name"] == "nastran.exe"]
            if not running:
                break
            time.sleep(0.5)

    def run(self):
        try:
            iteration_data_local = []  # Local copy
            path = self.gui.bdf_path.text()
            variables_str = self.gui.variables.text().strip()
            variables = [int(x.strip()) for x in variables_str.split(',') if x.strip()] if variables_str else []
            
            if len(variables) == 0:
                self.finished_signal.emit(False, "Must specify at least one variable")
                return  # Add early return
                
            result_type = self.gui.get_result_type()
            self.log_signal.emit(f"Loading BDF file: {path}")
            bdf = read_bdf(path)
            
            initial_mass = self.gui.get_mass(bdf)
            self.gui.initial_mass = initial_mass  # Store in GUI (reading is OK)
            if initial_mass is not None:
                self.log_signal.emit(f"Initial mass: {initial_mass:.2f}")
                self.mass_signal.emit(f"{initial_mass:.2f} (Initial)")  # USE SIGNAL!
            
            all_property_ids = list(bdf.properties.keys())
            self.log_signal.emit(f"Total properties in model: {len(all_property_ids)}")
            
            selected_property_ids = self.gui.parse_property_selection(
                self.gui.property_selection.text(), all_property_ids
            )
            self.log_signal.emit(f"Selected {len(selected_property_ids)} properties for optimization")
            
            original_values = {}
            property_ids = []
            for pid in selected_property_ids:
                if pid not in bdf.properties:
                    continue
                prop = bdf.properties[pid]
                property_ids.append(pid)
                if prop.type == "PSHELL":
                    original_values[pid] = ('PSHELL', prop.t)
                elif prop.type == "PCOMP":
                    original_values[pid] = ('PCOMP', prop.thicknesses[0])
                elif prop.type == "PBARL":
                    original_values[pid] = ('PBARL', prop.dim[0])
                else:
                    original_values[pid] = None
                    self.log_signal.emit(f"Warning: Property {pid} type {prop.type} is not supported")
            
            if not property_ids:
                raise ValueError("No valid properties selected for optimization")
            
            self.log_signal.emit(f"Optimizing {len(property_ids)} properties")
            self.log_signal.emit(f"Result type: {result_type.upper()}, Component: {self.gui.get_displacement_component().upper()}")
            
            iteration = [0]
            mode = self.gui.get_optimize_mode()
            best_result = [float('inf') if mode == 'minimize' else float('-inf')]
            best_multipliers = [None]
            best_mass = [None]
            history = []
            
            self.log_signal.emit("=" * 50)
            self.log_signal.emit("Starting optimization...")
            self.log_signal.emit("=" * 50)
            
            def objective_function(multipliers):
                if not self.gui.is_running:
                    raise StopIteration("Optimization stopped by user")
                
                iteration[0] += 1  # Always increment
                current_iter = iteration[0]
                
                try:
                    for i, pid in enumerate(property_ids):
                        if original_values[pid] is None:
                            continue
                        prop = bdf.properties[pid]
                        attr_name, original_value = original_values[pid]
                        if attr_name == 'PSHELL':
                            prop.t = original_value * multipliers[i]
                        elif attr_name == 'PCOMP':
                            prop.thicknesses[0] = original_value * multipliers[i]
                        elif attr_name == 'PBARL':
                            prop.dim[0] = original_value * multipliers[i]
                    
                    current_mass = self.gui.get_mass(bdf)
                    
                    if current_iter == 0:
                        bdf_name_new = "opt_initial.bdf"
                    else:
                        bdf_name_new = f"opt_{current_iter}.bdf"
                    
                    bdf.write_bdf(bdf_name_new)
                    
                    subprocess.call([self.gui.nastran_path.text(), bdf_name_new, "scr=yes"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.wait_for_nastran()  # Use thread's own method
                    
                    op2_name = bdf_name_new.replace(".bdf", ".op2")
                    variable_values = self.gui.extract_results_from_op2(op2_name, variables, result_type)
                    
                    if variable_values is None:
                        self.log_signal.emit(f"Failed to extract results in iteration {current_iter}")
                        return 1e10
                    
                    result = self.gui.evaluate_objective_function(
                        variable_values, self.gui.objective_function.text()
                    )
                    
                    if mode == 'minimize':
                        objective = result
                    elif mode == 'maximize':
                        objective = -result
                    elif mode == 'target':
                        objective = abs(result - float(self.gui.target_value.text()))
                    
                    objective = self.gui.apply_mass_penalty(objective, current_mass, mode)
                    
                    is_new_best = False
                    if mode == 'minimize' and result < best_result[0]:
                        best_result[0] = result
                        best_multipliers[0] = list(multipliers)
                        best_mass[0] = current_mass
                        self.gui.best_bdf_name = bdf_name_new
                        is_new_best = True
                  
                    elif mode == 'maximize' and result > best_result[0]:
                        best_result[0] = result
                        best_multipliers[0] = list(multipliers)
                        best_mass[0] = current_mass
                        self.gui.best_bdf_name = bdf_name_new
                        is_new_best = True
                     
                    elif mode == 'target' and abs(result - float(self.gui.target_value.text())) < abs(best_result[0] - float(self.gui.target_value.text())):
                        best_result[0] = result
                        best_multipliers[0] = list(multipliers)
                        best_mass[0] = current_mass
                        self.gui.best_bdf_name = bdf_name_new
                        is_new_best = True
                  
                    history.append({
                        'Iteration': current_iter, 
                        'Result': result, 
                        'Mass': current_mass if current_mass else 'N/A', 
                        **variable_values, 
                        'Multipliers': list(multipliers)
                    })

                    new_data = {
                    'iteration': current_iter, 
                    'result': result, 
                    'best_so_far': best_result[0], 
                    'mass': current_mass
                    }

                    iteration_data_local.append(new_data)

                                    
                    self.progress_signal.emit(current_iter, result, best_result[0], current_mass, is_new_best)
                    
                    if not is_new_best:
                        try:
                            for ext in [".f04", ".f06", ".log", ".op2", ".bdf"]:
                                os.remove(bdf_name_new.replace(".bdf", ext) if ext != ".bdf" else bdf_name_new)
                        except:
                            pass
                    
                    return objective
                except Exception as e:
                    self.log_signal.emit(f"ERROR in iteration {current_iter}: {e}")
                    return 1e10
            
            bounds = [(float(self.gui.min_bound.text()), float(self.gui.max_bound.text()))] * len(property_ids)
            method = self.gui.optimization_method.currentText()
            n_calls_val = int(self.gui.n_calls.text())
            
            self.log_signal.emit(f"Using optimization method: {method}")
            self.log_signal.emit(f"Target iterations: {n_calls_val}")
            
            if method == "Gaussian Process":
                n_initial = min(5, max(3, n_calls_val // 3))
                self.log_signal.emit(f"GP Minimize: n_calls={n_calls_val}, n_initial={n_initial}")
                result = gp_minimize(
                    objective_function,
                    bounds,
                    n_calls=n_calls_val,
                    n_initial_points=n_initial,
                    random_state=42,
                    verbose=False,
                    n_jobs=1
                )
                
            elif method == "Boosted Trees":
                n_initial = min(5, max(3, n_calls_val // 3))
                self.log_signal.emit(f"GBRT Minimize: n_calls={n_calls_val}, n_initial={n_initial}")
                result = gbrt_minimize(
                    objective_function,
                    bounds,
                    n_calls=n_calls_val,
                    n_initial_points=n_initial,
                    random_state=42,
                    verbose=False,
                    n_jobs=1
                )
                
            elif method == "Differential Evo":
                # Calculate DE parameters
                n_params = len(property_ids)
                popsize = 5 if n_params > 50 else 15
                
                # DE does popsize * maxiter * n_params evaluations
                maxiter = max(2, min(1000, n_calls_val // (popsize * n_params)))
                estimated_calls = popsize * maxiter * n_params
                
                self.log_signal.emit(f"Differential Evolution: popsize={popsize}, maxiter={maxiter}")
                self.log_signal.emit(f"Estimated function calls: {estimated_calls} (target: {n_calls_val})")
                
                def de_callback(xk, convergence):
                    if not self.gui.is_running:
                        self.log_signal.emit("Stopping: optimization halted by user")
                        return True
                    if iteration[0] >= n_calls_val:
                        self.log_signal.emit(f"Stopping: reached target of {n_calls_val} evaluations")
                        return True
                    return False
                
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=maxiter,
                    popsize=popsize,
                    seed=42,
                    polish=False,
                    workers=1,
                    callback=de_callback,
                    atol=0.001,
                    tol=0.01
                )
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            """
            if best_multipliers[0] is not None and self.gui.best_bdf_name:
                self.mesh_update_signal.emit(self.gui.best_bdf_name)
            """

            self.gui.iteration_data = iteration_data_local
            self.gui.save_results(property_ids, original_values, best_multipliers[0], 
                                history, best_result[0], best_mass[0])
            self.finished_signal.emit(True, "Optimization completed successfully")
            
        except StopIteration:
            self.finished_signal.emit(False, "Optimization stopped by user")
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.finished_signal.emit(False, error_msg)

def main():
    app = QApplication([])
    window = NastranOptimizerGUI()
    window.showMaximized()
    window.show()
    app.exec()

if __name__ == "__main__":
    main()