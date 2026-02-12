"""
Experiments Browser Widget for napari.

Browse, search, rate, and compare experiments from the BrainGlobe pipeline.
All detection, training, classification, and counting runs are logged here.
"""

import sys
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QFormLayout, QSpinBox, QTextEdit,
    QMessageBox, QSplitter, QAbstractItemView, QScrollArea
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

# Import paths from central config (auto-detects repo location)
_config_dir = Path(__file__).resolve().parent.parent.parent
if str(_config_dir) not in sys.path:
    sys.path.insert(0, str(_config_dir))
from mousebrain.config import BRAINS_ROOT, SCRIPTS_DIR

# Try to import experiment tracker
try:
    from experiment_tracker import ExperimentTracker, EXP_TYPES
    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    EXP_TYPES = ["detection", "training", "classification", "counts"]


class ExperimentsWidget(QWidget):
    """Widget for browsing and managing experiments."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.tracker = None
        self.current_experiments = []

        if TRACKER_AVAILABLE:
            try:
                self.tracker = ExperimentTracker()
            except Exception as e:
                print(f"Could not initialize tracker: {e}")

        self.setup_ui()
        self.refresh_experiments()

    def setup_ui(self):
        # Main layout with scroll area
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Title
        title = QLabel("Experiment Browser")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        if not TRACKER_AVAILABLE:
            warning = QLabel(
                "Experiment tracker not available.\n"
                "Make sure experiment_tracker.py is in the scripts folder."
            )
            warning.setStyleSheet("color: red;")
            layout.addWidget(warning)
            return

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Recent
        self.tabs.addTab(self.create_recent_tab(), "Recent")

        # Tab 2: Search
        self.tabs.addTab(self.create_search_tab(), "Search")

        # Tab 3: Best
        self.tabs.addTab(self.create_best_tab(), "Best Rated")

        # Tab 4: Statistics
        self.tabs.addTab(self.create_stats_tab(), "Statistics")

        # Tab 5: Compare
        self.tabs.addTab(self.create_compare_tab(), "Compare")

        # Detail panel at bottom
        detail_group = QGroupBox("Experiment Details")
        detail_layout = QVBoxLayout()
        detail_group.setLayout(detail_layout)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(200)
        detail_layout.addWidget(self.detail_text)

        # Rating buttons
        rating_layout = QHBoxLayout()
        rating_layout.addWidget(QLabel("Rate:"))

        self.rating_buttons = []
        for i in range(1, 6):
            btn = QPushButton(str(i))
            btn.setMaximumWidth(40)
            btn.clicked.connect(lambda checked, r=i: self.rate_current(r))
            rating_layout.addWidget(btn)
            self.rating_buttons.append(btn)

        rating_layout.addStretch()

        # View QC image button
        self.view_qc_btn = QPushButton("View QC Image")
        self.view_qc_btn.clicked.connect(self.view_qc_image)
        self.view_qc_btn.setEnabled(False)
        rating_layout.addWidget(self.view_qc_btn)

        detail_layout.addLayout(rating_layout)

        layout.addWidget(detail_group)

    def create_recent_tab(self):
        """Tab showing recent experiments."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Controls
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Show last:"))

        self.recent_limit = QSpinBox()
        self.recent_limit.setRange(5, 100)
        self.recent_limit.setValue(20)
        self.recent_limit.valueChanged.connect(self.refresh_experiments)
        ctrl_layout.addWidget(self.recent_limit)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_experiments)
        ctrl_layout.addWidget(refresh_btn)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Table
        self.recent_table = self.create_experiments_table()
        self.recent_table.itemClicked.connect(self.on_experiment_clicked)
        layout.addWidget(self.recent_table)

        return widget

    def create_search_tab(self):
        """Tab for searching experiments."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Search controls
        search_layout = QHBoxLayout()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Brain name, experiment ID...")
        self.search_input.returnPressed.connect(self.do_search)
        search_layout.addWidget(self.search_input)

        self.search_type = QComboBox()
        self.search_type.addItem("All types", None)
        for t in EXP_TYPES:
            self.search_type.addItem(t.capitalize(), t)
        search_layout.addWidget(self.search_type)

        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self.do_search)
        search_layout.addWidget(search_btn)

        layout.addLayout(search_layout)

        # Results table
        self.search_table = self.create_experiments_table()
        self.search_table.itemClicked.connect(self.on_experiment_clicked)
        layout.addWidget(self.search_table)

        return widget

    def create_best_tab(self):
        """Tab showing best-rated experiments."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))

        self.best_type = QComboBox()
        for t in EXP_TYPES:
            self.best_type.addItem(t.capitalize(), t)
        self.best_type.currentIndexChanged.connect(self.refresh_best)
        type_layout.addWidget(self.best_type)

        type_layout.addStretch()
        layout.addLayout(type_layout)

        # Table
        self.best_table = self.create_experiments_table()
        self.best_table.itemClicked.connect(self.on_experiment_clicked)
        layout.addWidget(self.best_table)

        return widget

    def create_stats_tab(self):
        """Tab showing statistics."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        layout.addWidget(self.stats_text)

        refresh_btn = QPushButton("Refresh Statistics")
        refresh_btn.clicked.connect(self.refresh_stats)
        layout.addWidget(refresh_btn)

        return widget

    def create_compare_tab(self):
        """Tab for comparing experiments."""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        info = QLabel(
            "Enter experiment IDs to compare (comma-separated):"
        )
        layout.addWidget(info)

        self.compare_input = QLineEdit()
        self.compare_input.setPlaceholderText("e.g., det_20250101_abc123, det_20250102_def456")
        layout.addWidget(self.compare_input)

        compare_btn = QPushButton("Compare")
        compare_btn.clicked.connect(self.do_compare)
        layout.addWidget(compare_btn)

        self.compare_table = QTableWidget()
        layout.addWidget(self.compare_table)

        return widget

    def create_experiments_table(self):
        """Create a table for displaying experiments."""
        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "Status", "Type", "ID", "Brain", "Rating", "Duration", "Date"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        return table

    def populate_table(self, table, rows):
        """Populate a table with experiment rows."""
        table.setRowCount(len(rows))

        status_symbols = {
            'completed': '\u2713',  # checkmark
            'failed': '\u2717',     # X
            'started': '\u25CB',    # circle
            'running': '\u25B6',    # play
        }

        for i, row in enumerate(rows):
            status = row.get('status', '?')
            symbol = status_symbols.get(status, '?')

            table.setItem(i, 0, QTableWidgetItem(symbol))
            table.setItem(i, 1, QTableWidgetItem(row.get('exp_type', '?')[:3].upper()))
            table.setItem(i, 2, QTableWidgetItem(row.get('exp_id', '?')))
            table.setItem(i, 3, QTableWidgetItem(row.get('brain', '?')))

            rating = row.get('rating', '')
            if rating:
                stars = '\u2605' * int(rating) + '\u2606' * (5 - int(rating))
            else:
                stars = '-----'
            table.setItem(i, 4, QTableWidgetItem(stars))

            duration = row.get('duration_seconds', '')
            if duration:
                try:
                    d = float(duration)
                    if d < 60:
                        dur_str = f"{d:.0f}s"
                    elif d < 3600:
                        dur_str = f"{d/60:.1f}m"
                    else:
                        dur_str = f"{d/3600:.1f}h"
                except:
                    dur_str = '-'
            else:
                dur_str = '-'
            table.setItem(i, 5, QTableWidgetItem(dur_str))

            created = row.get('created_at', '?')[:16]
            table.setItem(i, 6, QTableWidgetItem(created))

    def refresh_experiments(self):
        """Refresh the recent experiments list."""
        if not self.tracker:
            return

        limit = self.recent_limit.value()
        rows = self.tracker.get_recent(limit=limit)
        self.current_experiments = rows
        self.populate_table(self.recent_table, rows)
        self.refresh_stats()
        self.refresh_best()

    def do_search(self):
        """Perform a search."""
        if not self.tracker:
            return

        term = self.search_input.text()
        exp_type = self.search_type.currentData()

        rows = self.tracker.search(brain=term, exp_type=exp_type, limit=50)
        self.populate_table(self.search_table, rows)

    def refresh_best(self):
        """Refresh the best-rated list."""
        if not self.tracker:
            return

        exp_type = self.best_type.currentData()
        if exp_type:
            rows = self.tracker.get_best(exp_type, limit=10)
            self.populate_table(self.best_table, rows)

    def refresh_stats(self):
        """Refresh the statistics display."""
        if not self.tracker:
            return

        try:
            stats = self.tracker.get_statistics()

            html = "<h3>Experiment Statistics</h3>"
            html += f"<p><b>Total experiments:</b> {stats['total']}</p>"
            html += f"<p><b>Rated:</b> {stats['rated']}"
            if stats['avg_rating'] > 0:
                html += f" (avg: {stats['avg_rating']:.1f}\u2605)"
            html += "</p>"

            html += "<h4>By Type:</h4><ul>"
            for t, count in sorted(stats['by_type'].items()):
                html += f"<li>{t}: {count}</li>"
            html += "</ul>"

            html += "<h4>By Status:</h4><ul>"
            for s, count in sorted(stats['by_status'].items()):
                html += f"<li>{s}: {count}</li>"
            html += "</ul>"

            html += "<h4>Top Brains:</h4><ul>"
            sorted_brains = sorted(stats['by_brain'].items(), key=lambda x: x[1], reverse=True)[:5]
            for b, count in sorted_brains:
                html += f"<li>{b}: {count} experiments</li>"
            html += "</ul>"

            self.stats_text.setHtml(html)
        except Exception as e:
            self.stats_text.setText(f"Error loading stats: {e}")

    def on_experiment_clicked(self, item):
        """Handle experiment selection."""
        if not self.tracker:
            return

        table = item.tableWidget()
        row = item.row()
        exp_id = table.item(row, 2).text()

        exp = self.tracker.get_experiment(exp_id)
        if exp:
            self.show_experiment_detail(exp)

    def show_experiment_detail(self, exp):
        """Show detailed view of an experiment."""
        self.current_exp = exp

        html = f"<h3>{exp.get('exp_id', '?')}</h3>"
        html += f"<p><b>Type:</b> {exp.get('exp_type', '?')}</p>"
        html += f"<p><b>Brain:</b> {exp.get('brain', '?')}</p>"
        html += f"<p><b>Status:</b> {exp.get('status', '?')}</p>"
        html += f"<p><b>Created:</b> {exp.get('created_at', '?')}</p>"

        duration = exp.get('duration_seconds', '')
        if duration:
            try:
                d = float(duration)
                html += f"<p><b>Duration:</b> {d/60:.1f} minutes</p>"
            except:
                pass

        # Type-specific fields
        exp_type = exp.get('exp_type')
        if exp_type == 'detection':
            html += "<h4>Detection Parameters:</h4><ul>"
            html += f"<li>Preset: {exp.get('det_preset', '-')}</li>"
            html += f"<li>Ball XY: {exp.get('det_ball_xy', '-')}</li>"
            html += f"<li>Ball Z: {exp.get('det_ball_z', '-')}</li>"
            html += f"<li>Soma: {exp.get('det_soma_diameter', '-')}</li>"
            html += f"<li>Threshold: {exp.get('det_threshold', '-')}</li>"
            html += f"<li>Cells found: {exp.get('det_cells_found', '-')}</li>"
            html += "</ul>"
        elif exp_type == 'training':
            html += "<h4>Training Parameters:</h4><ul>"
            html += f"<li>Epochs: {exp.get('train_epochs', '-')}</li>"
            html += f"<li>Learning rate: {exp.get('train_learning_rate', '-')}</li>"
            html += f"<li>Augment: {exp.get('train_augment', '-')}</li>"
            html += f"<li>Loss: {exp.get('train_loss', '-')}</li>"
            html += f"<li>Accuracy: {exp.get('train_accuracy', '-')}</li>"
            html += "</ul>"
        elif exp_type == 'classification':
            html += "<h4>Classification Parameters:</h4><ul>"
            html += f"<li>Model: {exp.get('class_model_path', '-')}</li>"
            html += f"<li>Cells found: {exp.get('class_cells_found', '-')}</li>"
            html += f"<li>Rejected: {exp.get('class_rejected', '-')}</li>"
            html += "</ul>"
        elif exp_type == 'counts':
            html += "<h4>Counting Results:</h4><ul>"
            html += f"<li>Total cells: {exp.get('count_total_cells', '-')}</li>"
            html += f"<li>Output CSV: {exp.get('count_output_csv', '-')}</li>"
            html += "</ul>"

        notes = exp.get('notes', '')
        if notes:
            html += f"<p><b>Notes:</b> {notes}</p>"

        self.detail_text.setHtml(html)

        # Update rating buttons
        rating = exp.get('rating', '')
        for i, btn in enumerate(self.rating_buttons, 1):
            if rating and int(rating) == i:
                btn.setStyleSheet("background-color: gold;")
            else:
                btn.setStyleSheet("")

        # Update QC image button
        qc_path = exp.get('qc_image_path', '')
        self.view_qc_btn.setEnabled(bool(qc_path and Path(qc_path).exists()))

    def rate_current(self, rating):
        """Rate the currently selected experiment."""
        if not hasattr(self, 'current_exp') or not self.current_exp:
            return

        exp_id = self.current_exp.get('exp_id')
        if not exp_id:
            return

        try:
            self.tracker.rate_experiment(exp_id, rating)
            stars = '\u2605' * rating
            QMessageBox.information(self, "Rated", f"Rated {exp_id}: {stars}")
            self.refresh_experiments()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to rate: {e}")

    def do_compare(self):
        """Compare selected experiments."""
        if not self.tracker:
            return

        ids_text = self.compare_input.text()
        if not ids_text:
            return

        exp_ids = [x.strip() for x in ids_text.split(',')]
        rows = self.tracker.compare_experiments(exp_ids)

        if len(rows) < 2:
            QMessageBox.warning(self, "Error", "Could not find enough experiments to compare")
            return

        # Get all keys
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())

        # Filter to interesting keys
        skip_keys = {'hostname', 'script_version'}
        compare_keys = [k for k in sorted(all_keys)
                       if k not in skip_keys and any(row.get(k) for row in rows)]

        # Build comparison table
        self.compare_table.setColumnCount(len(rows) + 1)
        self.compare_table.setRowCount(len(compare_keys))

        headers = ["Field"] + [r.get('exp_id', '?')[:20] for r in rows]
        self.compare_table.setHorizontalHeaderLabels(headers)

        for i, key in enumerate(compare_keys):
            self.compare_table.setItem(i, 0, QTableWidgetItem(key))
            for j, row in enumerate(rows):
                val = str(row.get(key, ''))[:30]
                self.compare_table.setItem(i, j + 1, QTableWidgetItem(val))

        self.compare_table.resizeColumnsToContents()

    def view_qc_image(self):
        """Open the QC image for the currently selected experiment."""
        if not hasattr(self, 'current_exp') or not self.current_exp:
            return

        qc_path = self.current_exp.get('qc_image_path', '')
        if not qc_path:
            QMessageBox.warning(self, "No QC Image", "This experiment has no QC image.")
            return

        qc_path = Path(qc_path)
        if not qc_path.exists():
            QMessageBox.warning(self, "Not Found", f"QC image not found:\n{qc_path}")
            return

        import os
        import subprocess
        if os.name == 'nt':
            os.startfile(str(qc_path))
        else:
            subprocess.run(['open', str(qc_path)])
