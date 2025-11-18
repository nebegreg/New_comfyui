"""
ComfyUI Installer Widget for Mountain Studio Pro

PySide6 widget for managing ComfyUI models and custom nodes installation.
Provides user-friendly interface for:
- Path selection
- Model browsing and installation
- Custom node management
- Progress tracking
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QTableWidget, QTableWidgetItem,
    QProgressBar, QTextEdit, QCheckBox, QHeaderView, QMessageBox,
    QTabWidget, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor
import logging
from pathlib import Path
from typing import Optional

from core.ai.comfyui_installer import ComfyUIInstaller, ModelInfo, CustomNodeInfo

logger = logging.getLogger(__name__)


class InstallationThread(QThread):
    """Thread for background installation"""
    progress = Signal(float, float, float)  # current_mb, total_mb, percentage
    finished = Signal(bool, str)  # success, message
    log_message = Signal(str)

    def __init__(self, installer: ComfyUIInstaller, item_type: str, item):
        super().__init__()
        self.installer = installer
        self.item_type = item_type  # 'model' or 'node'
        self.item = item

    def run(self):
        try:
            if self.item_type == 'model':
                self.log_message.emit(f"Downloading {self.item.name}...")

                def progress_callback(current, total, pct):
                    self.progress.emit(current, total, pct)

                success = self.installer.download_model(self.item, progress_callback)

                if success:
                    self.finished.emit(True, f"Successfully installed {self.item.name}")
                else:
                    self.finished.emit(False, f"Failed to install {self.item.name}")

            elif self.item_type == 'node':
                self.log_message.emit(f"Installing {self.item.name}...")
                success = self.installer.install_custom_node(self.item)

                if success:
                    self.finished.emit(True, f"Successfully installed {self.item.name}")
                else:
                    self.finished.emit(False, f"Failed to install {self.item.name}")

        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")


class ComfyUIInstallerWidget(QWidget):
    """
    Widget for managing ComfyUI installation

    Features:
    - ComfyUI path selection and validation
    - Model installation with progress tracking
    - Custom node installation
    - Status overview
    - Batch installation for features
    """

    installation_complete = Signal()  # Emitted when installation finishes

    def __init__(self, parent=None):
        super().__init__(parent)

        self.installer = ComfyUIInstaller()
        self.install_thread: Optional[InstallationThread] = None

        self._init_ui()
        self._load_saved_path()

    def _init_ui(self):
        """Initialize UI components"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>🎨 ComfyUI Model & Node Manager</h2>")
        layout.addWidget(title)

        # Path selection section
        path_group = self._create_path_section()
        layout.addWidget(path_group)

        # Tabs for models and nodes
        tabs = QTabWidget()
        tabs.addTab(self._create_models_tab(), "📦 Models")
        tabs.addTab(self._create_nodes_tab(), "🔌 Custom Nodes")
        tabs.addTab(self._create_quick_install_tab(), "⚡ Quick Install")

        layout.addWidget(tabs)

        # Progress section
        progress_group = self._create_progress_section()
        layout.addWidget(progress_group)

        # Log section
        log_group = self._create_log_section()
        layout.addWidget(log_group)

    def _create_path_section(self) -> QGroupBox:
        """Create ComfyUI path selection section"""
        group = QGroupBox("ComfyUI Installation Path")
        layout = QHBoxLayout()

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Select your ComfyUI installation directory...")
        layout.addWidget(self.path_input)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_comfyui_path)
        layout.addWidget(browse_btn)

        self.validate_btn = QPushButton("Validate")
        self.validate_btn.clicked.connect(self._validate_path)
        layout.addWidget(self.validate_btn)

        self.path_status = QLabel("❌ Not set")
        layout.addWidget(self.path_status)

        group.setLayout(layout)
        return group

    def _create_models_tab(self) -> QWidget:
        """Create models management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info label
        info = QLabel("Recommended models for PBR texture and landscape generation:")
        layout.addWidget(info)

        # Models table
        self.models_table = QTableWidget()
        self.models_table.setColumnCount(5)
        self.models_table.setHorizontalHeaderLabels([
            "Model Name", "Size (MB)", "Status", "Description", "Action"
        ])

        header = self.models_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.models_table)

        # Buttons
        btn_layout = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.clicked.connect(self._refresh_models)
        btn_layout.addWidget(refresh_btn)

        install_all_btn = QPushButton("📥 Install All Models")
        install_all_btn.clicked.connect(self._install_all_models)
        btn_layout.addWidget(install_all_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # Populate table
        self._populate_models_table()

        return widget

    def _create_nodes_tab(self) -> QWidget:
        """Create custom nodes management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info label
        info = QLabel("Recommended custom nodes for PBR workflows:")
        layout.addWidget(info)

        # Nodes table
        self.nodes_table = QTableWidget()
        self.nodes_table.setColumnCount(4)
        self.nodes_table.setHorizontalHeaderLabels([
            "Node Name", "Status", "Description", "Action"
        ])

        header = self.nodes_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.nodes_table)

        # Buttons
        btn_layout = QHBoxLayout()

        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.clicked.connect(self._refresh_nodes)
        btn_layout.addWidget(refresh_btn)

        install_all_btn = QPushButton("📥 Install All Nodes")
        install_all_btn.clicked.connect(self._install_all_nodes)
        btn_layout.addWidget(install_all_btn)

        btn_layout.addStretch()

        layout.addLayout(btn_layout)

        # Populate table
        self._populate_nodes_table()

        return widget

    def _create_quick_install_tab(self) -> QWidget:
        """Create quick install tab for feature-based installation"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info = QLabel("<b>Quick Install for Specific Features</b><br>"
                     "Select features you want to use, and we'll install required components:")
        layout.addWidget(info)

        # Feature checkboxes
        self.feature_checkboxes = {}

        features = [
            ("pbr_textures", "PBR Texture Generation", "Generate diffuse, normal, roughness, AO maps"),
            ("landscape_generation", "Landscape Image Generation", "AI-generated landscape images"),
            ("normal_map_generation", "Normal Map Generation", "Generate normal maps from heightmaps"),
            ("high_quality_pbr", "High Quality PBR (SDXL)", "Use SDXL for highest quality (slower)"),
        ]

        for feature_id, name, description in features:
            checkbox = QCheckBox(f"<b>{name}</b> - {description}")
            self.feature_checkboxes[feature_id] = checkbox
            layout.addWidget(checkbox)

        # Install button
        install_btn = QPushButton("📥 Install Selected Features")
        install_btn.clicked.connect(self._install_selected_features)
        layout.addWidget(install_btn)

        layout.addStretch()

        return widget

    def _create_progress_section(self) -> QGroupBox:
        """Create progress tracking section"""
        group = QGroupBox("Installation Progress")
        layout = QVBoxLayout()

        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        group.setLayout(layout)
        return group

    def _create_log_section(self) -> QGroupBox:
        """Create log output section"""
        group = QGroupBox("Log")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        group.setLayout(layout)
        return group

    @Slot()
    def _browse_comfyui_path(self):
        """Browse for ComfyUI installation path"""
        path = QFileDialog.getExistingDirectory(
            self,
            "Select ComfyUI Installation Directory",
            str(Path.home())
        )

        if path:
            self.path_input.setText(path)
            self._validate_path()

    @Slot()
    def _validate_path(self):
        """Validate ComfyUI path"""
        path = self.path_input.text().strip()

        if not path:
            self.path_status.setText("❌ Not set")
            return

        if self.installer.set_comfyui_path(path):
            self.path_status.setText("✅ Valid")
            self.path_status.setStyleSheet("color: green;")
            self._log("✓ ComfyUI path validated")
            self._save_path(path)

            # Refresh tables
            self._refresh_models()
            self._refresh_nodes()
        else:
            self.path_status.setText("❌ Invalid")
            self.path_status.setStyleSheet("color: red;")
            self._log("✗ Invalid ComfyUI path")

    def _populate_models_table(self):
        """Populate models table with recommended models"""
        models = self.installer.get_recommended_models()
        self.models_table.setRowCount(len(models))

        for row, model in enumerate(models):
            # Name
            self.models_table.setItem(row, 0, QTableWidgetItem(model.name))

            # Size
            self.models_table.setItem(row, 1, QTableWidgetItem(str(model.size_mb)))

            # Status (will be updated by refresh)
            status_item = QTableWidgetItem("❓ Unknown")
            self.models_table.setItem(row, 2, status_item)

            # Description
            self.models_table.setItem(row, 3, QTableWidgetItem(model.description))

            # Action button
            install_btn = QPushButton("Install")
            install_btn.clicked.connect(lambda checked, m=model: self._install_model(m))
            self.models_table.setCellWidget(row, 4, install_btn)

        self._refresh_models()

    def _populate_nodes_table(self):
        """Populate nodes table with recommended custom nodes"""
        nodes = self.installer.get_recommended_custom_nodes()
        self.nodes_table.setRowCount(len(nodes))

        for row, node in enumerate(nodes):
            # Name
            self.nodes_table.setItem(row, 0, QTableWidgetItem(node.name))

            # Status (will be updated by refresh)
            status_item = QTableWidgetItem("❓ Unknown")
            self.nodes_table.setItem(row, 1, status_item)

            # Description
            self.nodes_table.setItem(row, 2, QTableWidgetItem(node.description))

            # Action button
            install_btn = QPushButton("Install")
            install_btn.clicked.connect(lambda checked, n=node: self._install_node(n))
            self.nodes_table.setCellWidget(row, 3, install_btn)

        self._refresh_nodes()

    @Slot()
    def _refresh_models(self):
        """Refresh model installation status"""
        models = self.installer.get_recommended_models()

        for row, model in enumerate(models):
            installed = self.installer.is_model_installed(model)

            status_item = self.models_table.item(row, 2)
            if installed:
                status_item.setText("✅ Installed")
                status_item.setForeground(QColor("green"))

                # Disable install button
                btn = self.models_table.cellWidget(row, 4)
                btn.setEnabled(False)
            else:
                status_item.setText("❌ Not installed")
                status_item.setForeground(QColor("red"))

                # Enable install button
                btn = self.models_table.cellWidget(row, 4)
                btn.setEnabled(True)

    @Slot()
    def _refresh_nodes(self):
        """Refresh custom node installation status"""
        nodes = self.installer.get_recommended_custom_nodes()

        for row, node in enumerate(nodes):
            installed = self.installer.is_node_installed(node)

            status_item = self.nodes_table.item(row, 1)
            if installed:
                status_item.setText("✅ Installed")
                status_item.setForeground(QColor("green"))

                # Disable install button
                btn = self.nodes_table.cellWidget(row, 3)
                btn.setEnabled(False)
            else:
                status_item.setText("❌ Not installed")
                status_item.setForeground(QColor("red"))

                # Enable install button
                btn = self.nodes_table.cellWidget(row, 3)
                btn.setEnabled(True)

    def _install_model(self, model: ModelInfo):
        """Install a specific model"""
        if not self.installer.comfyui_path:
            QMessageBox.warning(self, "Error", "Please set ComfyUI path first")
            return

        if self.install_thread and self.install_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Another installation is in progress")
            return

        self._log(f"Starting download: {model.name}")

        self.install_thread = InstallationThread(self.installer, 'model', model)
        self.install_thread.progress.connect(self._update_progress)
        self.install_thread.finished.connect(self._installation_finished)
        self.install_thread.log_message.connect(self._log)
        self.install_thread.start()

    def _install_node(self, node: CustomNodeInfo):
        """Install a specific custom node"""
        if not self.installer.comfyui_path:
            QMessageBox.warning(self, "Error", "Please set ComfyUI path first")
            return

        if self.install_thread and self.install_thread.isRunning():
            QMessageBox.warning(self, "Busy", "Another installation is in progress")
            return

        self._log(f"Starting installation: {node.name}")

        self.install_thread = InstallationThread(self.installer, 'node', node)
        self.install_thread.finished.connect(self._installation_finished)
        self.install_thread.log_message.connect(self._log)
        self.install_thread.start()

    @Slot()
    def _install_all_models(self):
        """Install all models (one by one)"""
        QMessageBox.information(
            self,
            "Batch Installation",
            "This will install all missing models one by one.\n"
            "This may take a while and require several GB of disk space.\n\n"
            "Installation will start after clicking OK."
        )

        # TODO: Implement queue-based batch installation
        self._log("Batch installation not yet implemented")

    @Slot()
    def _install_all_nodes(self):
        """Install all custom nodes"""
        QMessageBox.information(
            self,
            "Batch Installation",
            "This will install all missing custom nodes.\n"
            "Installation will start after clicking OK."
        )

        # TODO: Implement queue-based batch installation
        self._log("Batch installation not yet implemented")

    @Slot()
    def _install_selected_features(self):
        """Install components for selected features"""
        selected_features = [
            feature_id for feature_id, checkbox in self.feature_checkboxes.items()
            if checkbox.isChecked()
        ]

        if not selected_features:
            QMessageBox.warning(self, "No Selection", "Please select at least one feature")
            return

        if not self.installer.comfyui_path:
            QMessageBox.warning(self, "Error", "Please set ComfyUI path first")
            return

        self._log(f"Installing components for: {', '.join(selected_features)}")

        # Install required components
        successful, failed = self.installer.install_all_required(selected_features)

        self._log(f"Installation complete: {successful} successful, {failed} failed")
        self._refresh_models()
        self._refresh_nodes()

        QMessageBox.information(
            self,
            "Installation Complete",
            f"Installed {successful} components\nFailed: {failed}"
        )

    @Slot(float, float, float)
    def _update_progress(self, current_mb: float, total_mb: float, percentage: float):
        """Update progress bar"""
        self.progress_bar.setValue(int(percentage))
        self.progress_label.setText(f"Downloading: {current_mb:.1f} / {total_mb:.1f} MB ({percentage:.1f}%)")

    @Slot(bool, str)
    def _installation_finished(self, success: bool, message: str):
        """Handle installation completion"""
        self._log(message)

        if success:
            self.progress_bar.setValue(100)
            self.progress_label.setText("✓ Complete")

            # Refresh tables
            self._refresh_models()
            self._refresh_nodes()

            self.installation_complete.emit()
        else:
            self.progress_bar.setValue(0)
            self.progress_label.setText("✗ Failed")

    def _log(self, message: str):
        """Add message to log"""
        self.log_text.append(message)
        logger.info(message)

    def _save_path(self, path: str):
        """Save ComfyUI path to config"""
        config_file = Path.home() / '.mountain_studio' / 'comfyui_path.txt'
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(path)

    def _load_saved_path(self):
        """Load saved ComfyUI path"""
        config_file = Path.home() / '.mountain_studio' / 'comfyui_path.txt'

        if config_file.exists():
            path = config_file.read_text().strip()
            if path:
                self.path_input.setText(path)
                self._validate_path()


if __name__ == '__main__':
    # Test the widget
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)

    widget = ComfyUIInstallerWidget()
    widget.resize(900, 700)
    widget.show()

    sys.exit(app.exec())
