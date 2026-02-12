"""
Pipeline Dashboard Widget for napari.

A GUI for the entire BrainGlobe pipeline - see your brains, their status,
and run each step with a button click.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QGroupBox, QProgressBar,
    QComboBox, QMessageBox, QFileDialog, QCheckBox, QLineEdit,
    QWizard, QWizardPage, QTextEdit, QScrollArea
)
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtGui import QFont
import json

# Try to import BrainGlobe widgets (optional)
try:
    from brainreg.napari.register import brainreg_register
    BRAINREG_AVAILABLE = True
except Exception:
    # Catch all exceptions - includes ImportError and OSError from DLL failures
    BRAINREG_AVAILABLE = False

try:
    from cellfinder.napari.detect.detect import detect_widget
    from cellfinder.napari.train.train import training_widget
    CELLFINDER_AVAILABLE = True
except Exception:
    # Catch all exceptions - includes ImportError and OSError from torch DLL failures
    CELLFINDER_AVAILABLE = False

# Import paths from central config (auto-detects repo location)
_config_dir = Path(__file__).resolve().parent.parent.parent
if str(_config_dir) not in sys.path:
    sys.path.insert(0, str(_config_dir))
from mousebrain.config import BRAINS_ROOT, SCRIPTS_DIR


class SetupWizard(QWizard):
    """Wizard to help set up a new brain for the pipeline."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Brain to Pipeline")
        self.setWizardStyle(QWizard.ModernStyle)

        # Pages
        self.addPage(self.create_intro_page())
        self.addPage(self.create_file_page())
        self.addPage(self.create_naming_page())
        self.addPage(self.create_confirm_page())

        self.ims_file = None
        self.brain_name = None
        self.pipeline_dir = None  # Set after successful import
        self.start_extraction = False  # Whether to start extraction after wizard closes

    def create_intro_page(self):
        page = QWizardPage()
        page.setTitle("Getting Started")
        page.setSubTitle("This wizard will help you add a new brain to the pipeline.")

        layout = QVBoxLayout()

        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml("""
        <h3>What you need:</h3>
        <ul>
            <li>An <b>.ims file</b> from Imaris with your brain data</li>
            <li>The file should be named like: <code>349_CNT_01_02_1.625x_z4.ims</code></li>
        </ul>

        <h3>Naming convention:</h3>
        <p><code>NUMBER_PROJECT_COHORT_ANIMAL_MAGx_zSTEP.ims</code></p>
        <ul>
            <li><b>349</b> = Brain/sample number</li>
            <li><b>CNT</b> = Project code (CNT, SCI, etc.)</li>
            <li><b>01</b> = Cohort number</li>
            <li><b>02</b> = Animal number</li>
            <li><b>1.625x</b> = Magnification (used for voxel size!)</li>
            <li><b>z4</b> = Z-step in microns</li>
        </ul>

        <h3>What the wizard does:</h3>
        <ol>
            <li>Creates the correct folder structure</li>
            <li>Copies your .ims file to the right location</li>
            <li>Sets up metadata from the filename</li>
        </ol>
        """)
        layout.addWidget(info)

        page.setLayout(layout)
        return page

    def create_file_page(self):
        page = QWizardPage()
        page.setTitle("Select IMS File")
        page.setSubTitle("Choose the .ims file you want to process.")

        layout = QVBoxLayout()

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file selected...")
        layout.addWidget(self.file_path_edit)

        browse_btn = QPushButton("Browse for .ims file...")
        browse_btn.clicked.connect(self.browse_for_file)
        layout.addWidget(browse_btn)

        self.file_info = QLabel("")
        layout.addWidget(self.file_info)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def browse_for_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select IMS File", "",
            "Imaris Files (*.ims);;All Files (*)"
        )
        if file_path:
            self.ims_file = Path(file_path)
            self.file_path_edit.setText(str(self.ims_file))

            # Parse filename
            name = self.ims_file.stem
            self.file_info.setText(f"Filename: {name}")

            # Try to extract brain name
            # Expected: 349_CNT_01_02_1.625x_z4
            self.brain_name = name.replace(".", "p")  # dots to p for folders

    def create_naming_page(self):
        page = QWizardPage()
        page.setTitle("Confirm Names")
        page.setSubTitle("Verify the brain name and mouse folder.")

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Mouse folder name:"))
        self.mouse_edit = QLineEdit()
        self.mouse_edit.setPlaceholderText("e.g., 349_CNT_01_02")
        layout.addWidget(self.mouse_edit)

        layout.addWidget(QLabel("Pipeline folder name:"))
        self.pipeline_edit = QLineEdit()
        self.pipeline_edit.setPlaceholderText("e.g., 349_CNT_01_02_1p625x_z4")
        layout.addWidget(self.pipeline_edit)

        layout.addWidget(QLabel(""))

        info = QLabel(
            "The mouse folder groups all acquisitions of the same animal.\n"
            "The pipeline folder is for this specific magnification/z-step."
        )
        info.setStyleSheet("color: gray;")
        layout.addWidget(info)

        layout.addStretch()
        page.setLayout(layout)
        return page

    def initializePage(self, page_id):
        """Called when entering a page."""
        super().initializePage(page_id)

        # When entering naming page, auto-fill based on filename
        if page_id == 2 and self.brain_name:
            # Try to extract mouse name (first 4 parts)
            parts = self.brain_name.split("_")
            if len(parts) >= 4:
                mouse_name = "_".join(parts[:4])
                self.mouse_edit.setText(mouse_name)
            self.pipeline_edit.setText(self.brain_name)

    def create_confirm_page(self):
        page = QWizardPage()
        page.setTitle("Confirm Setup")
        page.setSubTitle("Review and confirm the folder structure.")

        layout = QVBoxLayout()

        self.confirm_text = QTextEdit()
        self.confirm_text.setReadOnly(True)
        layout.addWidget(self.confirm_text)

        page.setLayout(layout)
        return page

    def validateCurrentPage(self):
        """Validate before moving to next page."""
        current = self.currentId()

        if current == 1:  # File page
            if not self.ims_file or not self.ims_file.exists():
                QMessageBox.warning(self, "Error", "Please select a valid .ims file")
                return False

        if current == 2:  # Naming page
            if not self.mouse_edit.text() or not self.pipeline_edit.text():
                QMessageBox.warning(self, "Error", "Please fill in both names")
                return False

            # Update confirm text
            mouse_name = self.mouse_edit.text()
            pipeline_name = self.pipeline_edit.text()

            self.confirm_text.setHtml(f"""
            <h3>Will create:</h3>
            <pre>
{BRAINS_ROOT}/
    {mouse_name}/
        {pipeline_name}/
            0_Raw_IMS/
                {self.ims_file.name}
            1_Extracted_Full/
            2_Cropped_For_Registration/
            3_Registered_Atlas/
            4_Cell_Candidates/
            5_Classified_Cells/
            6_Region_Analysis/
            </pre>

            <p><b>Source file:</b> {self.ims_file}</p>
            <p><b>File size:</b> {self.ims_file.stat().st_size / (1024**3):.2f} GB</p>
            """)

        return True

    def accept(self):
        """Create the folder structure and copy the file."""
        mouse_name = self.mouse_edit.text()
        pipeline_name = self.pipeline_edit.text()

        mouse_dir = BRAINS_ROOT / mouse_name
        pipeline_dir = mouse_dir / pipeline_name

        try:
            # Create folders
            folders = [
                "0_Raw_IMS",
                "1_Extracted_Full",
                "2_Cropped_For_Registration",
                "3_Registered_Atlas",
                "4_Cell_Candidates",
                "5_Classified_Cells",
                "6_Region_Analysis",
            ]

            for folder in folders:
                (pipeline_dir / folder).mkdir(parents=True, exist_ok=True)

            # Copy IMS file
            dest = pipeline_dir / "0_Raw_IMS" / self.ims_file.name
            if not dest.exists():
                QMessageBox.information(
                    self, "Copying File",
                    f"Copying {self.ims_file.name}...\nThis may take a while for large files."
                )
                shutil.copy2(self.ims_file, dest)

            # Store the pipeline directory for extraction
            self.pipeline_dir = pipeline_dir

            # Ask if user wants to start extraction
            reply = QMessageBox.question(
                self, "Start Extraction?",
                f"Brain '{pipeline_name}' has been set up!\n\n"
                f"Would you like to start extracting images now?\n"
                f"(This will take 20-40 minutes for large files)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            self.start_extraction = (reply == QMessageBox.Yes)

            super().accept()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set up brain:\n{e}")


class ScriptRunner(QThread):
    """Run a script in a background thread."""
    finished = Signal(bool, str)
    output = Signal(str)

    def __init__(self, script_name, args=None):
        super().__init__()
        self.script_name = script_name
        self.args = args or []

    def run(self):
        script_path = SCRIPTS_DIR / self.script_name
        cmd = [sys.executable, str(script_path)] + self.args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(SCRIPTS_DIR)
            )
            success = result.returncode == 0
            output = result.stdout + result.stderr
            self.finished.emit(success, output)
        except Exception as e:
            self.finished.emit(False, str(e))


class PipelineDashboard(QWidget):
    """Main pipeline dashboard widget."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.current_brain = None
        self.runner = None

        self.setup_ui()
        self.refresh_brains()

    def setup_ui(self):
        # Main layout with scroll area
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Content widget inside scroll area
        content = QWidget()
        layout = QVBoxLayout()
        content.setLayout(layout)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Title
        title = QLabel("Pipeline Dashboard")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)

        # Brain list
        brain_group = QGroupBox("Your Brains")
        brain_layout = QVBoxLayout()
        brain_group.setLayout(brain_layout)

        self.brain_list = QListWidget()
        self.brain_list.itemClicked.connect(self.on_brain_selected)
        brain_layout.addWidget(self.brain_list)

        btn_row = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_brains)
        btn_row.addWidget(refresh_btn)

        add_brain_btn = QPushButton("+ Add New Brain")
        add_brain_btn.clicked.connect(self.add_new_brain)
        add_brain_btn.setStyleSheet("background-color: #2196F3; color: white;")
        btn_row.addWidget(add_brain_btn)

        brain_layout.addLayout(btn_row)

        layout.addWidget(brain_group)

        # Status group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)

        self.status_label = QLabel("Select a brain above")
        status_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(7)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)

        # Checklist
        self.check_labels = []
        checks = [
            "IMS file present",
            "Images extracted",
            "Cropped for registration",
            "Registered to atlas",
            "Registration approved",
            "Cells detected",
            "Cells classified",
            "Regions counted",
        ]
        for check in checks:
            lbl = QLabel(f"[ ] {check}")
            self.check_labels.append(lbl)
            status_layout.addWidget(lbl)

        layout.addWidget(status_group)

        # Actions group
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()
        action_group.setLayout(action_layout)

        self.next_step_label = QLabel("Next step: -")
        self.next_step_label.setFont(QFont("Arial", 10, QFont.Bold))
        action_layout.addWidget(self.next_step_label)

        self.run_next_btn = QPushButton("Run Next Step")
        self.run_next_btn.clicked.connect(self.run_next_step)
        self.run_next_btn.setEnabled(False)
        self.run_next_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        action_layout.addWidget(self.run_next_btn)

        # Detection preset
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Detection preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["balanced", "sensitive", "conservative", "large_cells"])
        preset_layout.addWidget(self.preset_combo)
        action_layout.addLayout(preset_layout)

        # Manual actions
        action_layout.addWidget(QLabel("Manual actions:"))

        btn_layout = QHBoxLayout()

        self.crop_btn = QPushButton("Manual Crop")
        self.crop_btn.clicked.connect(self.open_manual_crop)
        self.crop_btn.setEnabled(False)
        btn_layout.addWidget(self.crop_btn)

        self.view_qc_btn = QPushButton("View QC")
        self.view_qc_btn.clicked.connect(self.view_qc_image)
        self.view_qc_btn.setEnabled(False)
        btn_layout.addWidget(self.view_qc_btn)

        action_layout.addLayout(btn_layout)

        btn_layout2 = QHBoxLayout()

        self.approve_btn = QPushButton("Approve Registration")
        self.approve_btn.clicked.connect(self.approve_registration)
        self.approve_btn.setEnabled(False)
        btn_layout2.addWidget(self.approve_btn)

        self.view_results_btn = QPushButton("View Results")
        self.view_results_btn.clicked.connect(self.view_results)
        self.view_results_btn.setEnabled(False)
        btn_layout2.addWidget(self.view_results_btn)

        action_layout.addLayout(btn_layout2)

        layout.addWidget(action_group)

        # BrainGlobe Tools group
        bg_group = QGroupBox("BrainGlobe Tools")
        bg_layout = QVBoxLayout()
        bg_group.setLayout(bg_layout)

        bg_info = QLabel("Launch native BrainGlobe widgets with your settings:")
        bg_info.setStyleSheet("color: gray; font-size: 10px;")
        bg_layout.addWidget(bg_info)

        bg_btn_layout = QHBoxLayout()

        self.brainreg_btn = QPushButton("Registration")
        self.brainreg_btn.clicked.connect(self.launch_brainreg)
        self.brainreg_btn.setEnabled(BRAINREG_AVAILABLE)
        self.brainreg_btn.setToolTip("Launch brainreg with your voxel settings")
        bg_btn_layout.addWidget(self.brainreg_btn)

        self.cellfinder_btn = QPushButton("Cell Detection")
        self.cellfinder_btn.clicked.connect(self.launch_cellfinder_detect)
        self.cellfinder_btn.setEnabled(CELLFINDER_AVAILABLE)
        self.cellfinder_btn.setToolTip("Launch cellfinder detection with your settings")
        bg_btn_layout.addWidget(self.cellfinder_btn)

        self.training_btn = QPushButton("Training")
        self.training_btn.clicked.connect(self.launch_cellfinder_train)
        self.training_btn.setEnabled(CELLFINDER_AVAILABLE)
        self.training_btn.setToolTip("Launch cellfinder network training")
        bg_btn_layout.addWidget(self.training_btn)

        bg_layout.addLayout(bg_btn_layout)

        if not BRAINREG_AVAILABLE:
            bg_layout.addWidget(QLabel("brainreg not installed"))
        if not CELLFINDER_AVAILABLE:
            bg_layout.addWidget(QLabel("cellfinder not installed"))

        layout.addWidget(bg_group)

        # Status message
        self.message_label = QLabel("")
        layout.addWidget(self.message_label)

        # Add stretch to push everything up
        layout.addStretch()

    def refresh_brains(self):
        """Refresh the brain list."""
        self.brain_list.clear()

        if not BRAINS_ROOT.exists():
            self.brain_list.addItem("Brains folder not found!")
            return

        for mouse_dir in sorted(BRAINS_ROOT.iterdir()):
            if not mouse_dir.is_dir() or mouse_dir.name.startswith('.'):
                continue
            if any(skip in mouse_dir.name.lower() for skip in ['script', 'backup', 'archive', 'summary']):
                continue

            for pipeline_dir in sorted(mouse_dir.iterdir()):
                if not pipeline_dir.is_dir():
                    continue

                # Check if this looks like a pipeline
                has_structure = any([
                    (pipeline_dir / "0_Raw_IMS").exists(),
                    (pipeline_dir / "1_Extracted_Full").exists(),
                    (pipeline_dir / "2_Cropped_For_Registration").exists(),
                ])

                if has_structure:
                    status = self.get_brain_status(pipeline_dir)
                    progress = status['step_num'] - 1

                    # Create item with progress indicator
                    bar = "=" * progress + ">" + " " * (7 - progress) if progress < 7 else "======="
                    text = f"{pipeline_dir.name}  [{bar}]"

                    item = QListWidgetItem(text)
                    item.setData(Qt.UserRole, str(pipeline_dir))
                    self.brain_list.addItem(item)

    def get_brain_status(self, brain_path):
        """Get status of a brain."""
        p = Path(brain_path)

        status = {
            'has_ims': len(list((p / "0_Raw_IMS").glob("*.ims"))) > 0 if (p / "0_Raw_IMS").exists() else False,
            'extracted': (p / "1_Extracted_Full" / "metadata.json").exists() if (p / "1_Extracted_Full").exists() else False,
            'cropped': (p / "2_Cropped_For_Registration" / "metadata.json").exists() if (p / "2_Cropped_For_Registration").exists() else False,
            'registered': (p / "3_Registered_Atlas" / "brainreg.json").exists() if (p / "3_Registered_Atlas").exists() else False,
            'approved': (p / "3_Registered_Atlas" / ".registration_approved").exists() if (p / "3_Registered_Atlas").exists() else False,
            'detected': len(list((p / "4_Cell_Candidates").glob("*.xml"))) > 0 if (p / "4_Cell_Candidates").exists() else False,
            'classified': (p / "5_Classified_Cells" / "cells.xml").exists() if (p / "5_Classified_Cells").exists() else False,
            'counted': len(list((p / "6_Region_Analysis").glob("*.csv"))) > 0 if (p / "6_Region_Analysis").exists() else False,
        }

        # Determine current step
        if not status['extracted']:
            status['step'] = 'extract'
            status['step_name'] = 'Extract images'
            status['step_num'] = 1
        elif not status['cropped']:
            status['step'] = 'crop'
            status['step_name'] = 'Crop spinal cord'
            status['step_num'] = 2
        elif not status['registered']:
            status['step'] = 'register'
            status['step_name'] = 'Register to atlas'
            status['step_num'] = 3
        elif not status['approved']:
            status['step'] = 'approve'
            status['step_name'] = 'Approve registration'
            status['step_num'] = 4
        elif not status['detected']:
            status['step'] = 'detect'
            status['step_name'] = 'Detect cells'
            status['step_num'] = 5
        elif not status['classified']:
            status['step'] = 'classify'
            status['step_name'] = 'Classify cells'
            status['step_num'] = 6
        elif not status['counted']:
            status['step'] = 'count'
            status['step_name'] = 'Count regions'
            status['step_num'] = 7
        else:
            status['step'] = 'done'
            status['step_name'] = 'Complete!'
            status['step_num'] = 8

        return status

    def add_new_brain(self):
        """Launch the setup wizard to add a new brain."""
        wizard = SetupWizard(self)
        if wizard.exec_():
            self.refresh_brains()

            # If user requested extraction, select the brain and run it
            if wizard.start_extraction and wizard.pipeline_dir:
                # Find and select the new brain in the list
                for i in range(self.brain_list.count()):
                    item = self.brain_list.item(i)
                    if str(wizard.pipeline_dir) == item.data(Qt.UserRole):
                        self.brain_list.setCurrentItem(item)
                        self.on_brain_selected(item)
                        # Start extraction
                        self.run_script('2_extract_and_analyze.py', [])

    def on_brain_selected(self, item):
        """Handle brain selection."""
        brain_path = Path(item.data(Qt.UserRole))
        self.current_brain = brain_path
        self.update_status_display()

    def update_status_display(self):
        """Update the status display for current brain."""
        if not self.current_brain:
            return

        status = self.get_brain_status(self.current_brain)

        # Update progress bar
        self.progress_bar.setValue(status['step_num'] - 1)

        # Update status label
        self.status_label.setText(f"Step {status['step_num']}/7: {status['step_name']}")

        # Update checkmarks
        checks = [
            status['has_ims'],
            status['extracted'],
            status['cropped'],
            status['registered'],
            status['approved'],
            status['detected'],
            status['classified'],
            status['counted'],
        ]

        labels = [
            "IMS file present",
            "Images extracted",
            "Cropped for registration",
            "Registered to atlas",
            "Registration approved",
            "Cells detected",
            "Cells classified",
            "Regions counted",
        ]

        for lbl, done, text in zip(self.check_labels, checks, labels):
            mark = "[X]" if done else "[ ]"
            lbl.setText(f"{mark} {text}")

        # Update next step
        self.next_step_label.setText(f"Next: {status['step_name']}")

        # Enable/disable buttons
        self.run_next_btn.setEnabled(status['step'] != 'done')
        self.crop_btn.setEnabled(status['extracted'])
        self.view_qc_btn.setEnabled(status['registered'])
        self.approve_btn.setEnabled(status['registered'] and not status['approved'])
        self.view_results_btn.setEnabled(status['counted'])

    def run_next_step(self):
        """Run the next step in the pipeline."""
        if not self.current_brain:
            return

        status = self.get_brain_status(self.current_brain)
        brain_name = self.current_brain.name

        self.message_label.setText(f"Running {status['step_name']}...")
        self.run_next_btn.setEnabled(False)

        if status['step'] == 'extract':
            self.run_script('2_extract_and_analyze.py', [])

        elif status['step'] == 'crop':
            # Open manual crop widget
            self.open_manual_crop()
            self.message_label.setText("Use Manual Crop widget, then refresh")
            self.run_next_btn.setEnabled(True)

        elif status['step'] == 'register':
            self.run_script('3_register_to_atlas.py', ['--batch'])

        elif status['step'] == 'approve':
            # Show QC first
            self.view_qc_image()
            reply = QMessageBox.question(
                self, 'Approve Registration',
                'Does the registration look good?',
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.run_script('util_approve_registration.py', ['--brain', brain_name, '--yes'])
            else:
                self.message_label.setText("Registration not approved. Re-register if needed.")
                self.run_next_btn.setEnabled(True)

        elif status['step'] == 'detect':
            preset = self.preset_combo.currentText()
            self.run_script('4_detect_cells.py', ['--brain', brain_name, '--preset', preset])

        elif status['step'] == 'classify':
            self.run_script('5_classify_cells.py', ['--brain', brain_name])

        elif status['step'] == 'count':
            self.run_script('6_count_regions.py', ['--brain', brain_name])

    def run_script(self, script_name, args):
        """Run a script in background thread."""
        self.runner = ScriptRunner(script_name, args)
        self.runner.finished.connect(self.on_script_finished)
        self.runner.start()

    def on_script_finished(self, success, output):
        """Handle script completion."""
        if success:
            self.message_label.setText("Done!")
        else:
            self.message_label.setText("Error - check console")
            print(output)

        self.refresh_brains()
        self.update_status_display()
        self.run_next_btn.setEnabled(True)

    def open_manual_crop(self):
        """Open the manual crop widget."""
        # The manual crop widget should already be available in napari
        self.message_label.setText("Use Plugins > Manual Crop Tool")

    def view_qc_image(self):
        """View the QC image."""
        if not self.current_brain:
            return

        reg_folder = self.current_brain / "3_Registered_Atlas"

        # Try multiple possible QC image names
        qc_candidates = [
            "QC_registration_detailed.png",
            "QC_registration_overview.png",
            "qc_registration.png",
        ]

        qc_path = None
        for candidate in qc_candidates:
            path = reg_folder / candidate
            if path.exists():
                qc_path = path
                break

        if qc_path:
            import os
            if os.name == 'nt':
                os.startfile(str(qc_path))
            else:
                subprocess.run(['open', str(qc_path)])
        else:
            QMessageBox.warning(self, "Not Found", f"No QC image found in:\n{reg_folder}")

    def approve_registration(self):
        """Approve the registration."""
        if not self.current_brain:
            return

        reply = QMessageBox.question(
            self, 'Approve Registration',
            'Have you reviewed the QC image?\nDoes the registration look good?',
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            brain_name = self.current_brain.name
            self.run_script('util_approve_registration.py', ['--brain', brain_name, '--yes'])

    def view_results(self):
        """View the results folder."""
        if not self.current_brain:
            return

        results_path = self.current_brain / "6_Region_Analysis"
        if results_path.exists():
            import os
            if os.name == 'nt':
                os.startfile(str(results_path))
            else:
                subprocess.run(['open', str(results_path)])

    def get_brain_metadata(self):
        """Get metadata from current brain for pre-filling widgets."""
        if not self.current_brain:
            return {}

        metadata = {}

        # Try to load from extracted metadata
        for folder in ["1_Extracted_Full", "2_Cropped_For_Registration"]:
            meta_path = self.current_brain / folder / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        data = json.load(f)
                        metadata.update(data)
                except:
                    pass

        # Try to parse from brain name (e.g., 349_CNT_01_02_1p625x_z4)
        name = self.current_brain.name
        parts = name.split("_")

        # Look for magnification (e.g., 1p625x)
        for part in parts:
            if "x" in part.lower() and "p" in part.lower():
                try:
                    mag_str = part.lower().replace("x", "").replace("p", ".")
                    mag = float(mag_str)
                    metadata.setdefault('magnification', mag)
                    # Calculate voxel size: camera_pixel / magnification
                    camera_pixel = 6.5  # Andor Neo/Zyla default
                    voxel_xy = camera_pixel / mag
                    metadata.setdefault('voxel_xy', voxel_xy)
                except:
                    pass

            # Look for z-step (e.g., z4)
            if part.lower().startswith("z") and part[1:].isdigit():
                try:
                    z_step = float(part[1:])
                    metadata.setdefault('voxel_z', z_step)
                except:
                    pass

        return metadata

    def launch_brainreg(self):
        """Launch brainreg widget with pre-filled settings."""
        if not BRAINREG_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "brainreg is not installed")
            return

        try:
            # Create the widget
            brainreg_w = brainreg_register()

            # Pre-fill with metadata
            metadata = self.get_brain_metadata()

            if 'voxel_z' in metadata:
                brainreg_w.z_pixel_um.value = metadata['voxel_z']
            if 'voxel_xy' in metadata:
                brainreg_w.y_pixel_um.value = metadata['voxel_xy']
                brainreg_w.x_pixel_um.value = metadata['voxel_xy']

            # Set default orientation (posterior-superior-left for our data)
            brainreg_w.data_orientation.value = "psl"

            # Set output folder if brain is selected
            if self.current_brain:
                output_dir = self.current_brain / "3_Registered_Atlas"
                brainreg_w.registration_output_folder.value = output_dir

            # Set reasonable defaults for registration
            brainreg_w.n_free_cpus.value = 2

            # Add to napari viewer
            self.viewer.window.add_dock_widget(
                brainreg_w, area='right', name='Brain Registration'
            )

            self.message_label.setText("Opened brainreg widget")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to launch brainreg:\n{e}")

    def launch_cellfinder_detect(self):
        """Launch cellfinder detection widget with pre-filled settings."""
        if not CELLFINDER_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "cellfinder is not installed")
            return

        try:
            # Create the widget
            detect_w = detect_widget()

            # Pre-fill with metadata
            metadata = self.get_brain_metadata()

            if 'voxel_z' in metadata:
                detect_w.voxel_size_z.value = metadata['voxel_z']
            if 'voxel_xy' in metadata:
                detect_w.voxel_size_y.value = metadata['voxel_xy']
                detect_w.voxel_size_x.value = metadata['voxel_xy']

            # Get preset settings from the combo box
            preset = self.preset_combo.currentText()
            presets = {
                'sensitive': {'ball_xy': 4, 'ball_z': 10, 'soma': 12, 'threshold': 8},
                'balanced': {'ball_xy': 6, 'ball_z': 15, 'soma': 16, 'threshold': 10},
                'conservative': {'ball_xy': 8, 'ball_z': 20, 'soma': 20, 'threshold': 12},
                'large_cells': {'ball_xy': 10, 'ball_z': 25, 'soma': 25, 'threshold': 10},
            }

            if preset in presets:
                p = presets[preset]
                detect_w.ball_xy_size.value = p['ball_xy']
                detect_w.ball_z_size.value = p['ball_z']
                detect_w.soma_diameter.value = float(p['soma'])
                detect_w.n_sds_above_mean_thresh.value = p['threshold']

            # Use pre-trained weights by default
            detect_w.use_pre_trained_weights.value = True
            detect_w.n_free_cpus.value = 2

            # Add to napari viewer
            self.viewer.window.add_dock_widget(
                detect_w, area='right', name='Cell Detection'
            )

            self.message_label.setText("Opened cellfinder detection widget")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to launch cellfinder:\n{e}")

    def launch_cellfinder_train(self):
        """Launch cellfinder training widget."""
        if not CELLFINDER_AVAILABLE:
            QMessageBox.warning(self, "Not Available", "cellfinder is not installed")
            return

        try:
            # Create the widget
            train_w = training_widget()

            # Set reasonable defaults
            train_w.epochs.value = 100
            train_w.learning_rate.value = 1e-4
            train_w.batch_size.value = 16
            train_w.augment.value = True
            train_w.save_progress.value = True
            train_w.number_of_free_cpus.value = 2

            # Add to napari viewer
            self.viewer.window.add_dock_widget(
                train_w, area='right', name='Network Training'
            )

            self.message_label.setText("Opened cellfinder training widget")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to launch training widget:\n{e}")
