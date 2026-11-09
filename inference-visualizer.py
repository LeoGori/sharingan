#!/usr/bin/env python3
"""Side-by-side image comparison GUI.

Shows, for each image filename shared across the input folders, the source
image next to its processed/modified counterparts, so you can flip through a
dataset and visually compare inference outputs against the originals (or
against each other).

Usage:
    python3 inference-visualizer.py --source-dir /path/to/original \\
        [--processed-dirs /path/to/processed1 [/path/to/processed2 /path/to/processed3]]

Up to 4 folders total are supported (the source folder plus up to 3
--processed-dirs). --processed-dirs is optional; omitting it just browses
--source-dir on its own. Images are matched by filename across all folders;
only filenames present in every folder are shown, and the terminal prints a
warning listing anything skipped.

Controls:
    - Left / Right arrow keys, or the Previous / Next buttons: step through
      images.
    - Home / End: jump to the first / last image.
    - Click a filename in the left sidebar to jump straight to it; the
      filter box above it narrows the list.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
MAX_TOTAL_FOLDERS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Side-by-side viewer for comparing an original image folder "
            "against one or more processed/modified versions of the same images."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Folder containing the original images.",
    )
    parser.add_argument(
        "--processed-dirs",
        type=Path,
        nargs="+",
        default=[],
        metavar="DIR",
        help=(
            f"Zero to {MAX_TOTAL_FOLDERS - 1} folders containing modified versions "
            "of the images in --source-dir, matched by filename. If omitted, only "
            "--source-dir is shown."
        ),
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        parser.error(f"--source-dir is not a directory: {args.source_dir}")

    if len(args.processed_dirs) > MAX_TOTAL_FOLDERS - 1:
        parser.error(
            f"too many --processed-dirs ({len(args.processed_dirs)}); the "
            f"maximum is {MAX_TOTAL_FOLDERS - 1} (source dir + up to "
            f"{MAX_TOTAL_FOLDERS - 1} processed dirs = {MAX_TOTAL_FOLDERS} total)."
        )

    for d in args.processed_dirs:
        if not d.is_dir():
            parser.error(f"--processed-dirs entry is not a directory: {d}")

    return args


def list_images(folder: Path) -> dict[str, Path]:
    """Return {filename: path} for image files directly inside `folder`."""
    images = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images[p.name] = p
    return images


def common_filenames(folder_images: list[dict[str, Path]]) -> list[str]:
    common = set(folder_images[0].keys())
    for images in folder_images[1:]:
        common &= set(images.keys())
    return sorted(common)


def report_mismatches(folders: list[Path], folder_images: list[dict[str, Path]], common: set[str]) -> None:
    for folder, images in zip(folders, folder_images):
        missing = sorted(set(images.keys()) - common)
        if missing:
            preview = ", ".join(missing[:10])
            more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
            print(f"[warn] {folder}: {len(missing)} image(s) not shown (missing from other folders): {preview}{more}")


class ImagePanel(QFrame):
    """A titled panel that shows one image, scaled to fit, keeping aspect ratio."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)

        self._pixmap: QPixmap | None = None

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-weight: bold; padding: 4px;")

        self.image_label = QLabel("(no image)")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1, 1)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setStyleSheet("background-color: #202020; color: #aaaaaa;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.title_label)
        layout.addWidget(self.image_label, stretch=1)

    def set_image(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            self._pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("(missing)")
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("(failed to load)")
            return

        self._pixmap = pixmap
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()


class MainWindow(QMainWindow):
    def __init__(self, source_dir: Path, processed_dirs: list[Path]):
        super().__init__()
        self.setWindowTitle("Inference Visualizer")
        self.resize(1400, 900)

        self.folders = [source_dir] + processed_dirs
        self.folder_images = [list_images(f) for f in self.folders]
        self.filenames = common_filenames(self.folder_images)
        report_mismatches(self.folders, self.folder_images, set(self.filenames))

        self.current_index = 0
        self._filtered = self.filenames

        self._build_ui()
        self._show_current()

        if not self.filenames:
            QMessageBox.warning(
                self,
                "No common images",
                "No image files with matching names were found across all provided folders.",
            )

    def _folder_title(self, index: int, folder: Path) -> str:
        prefix = "source" if index == 0 else f"processed {index}"
        return f"{prefix}: {folder}"

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        # --- Sidebar: filter box + filename list ---
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(240)
        sidebar = QVBoxLayout(sidebar_widget)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter filenames...")
        self.filter_edit.textChanged.connect(self._apply_filter)
        sidebar.addWidget(self.filter_edit)

        self.list_widget = QListWidget()
        self.list_widget.addItems(self.filenames)
        self.list_widget.currentRowChanged.connect(self._on_list_selection)
        sidebar.addWidget(self.list_widget, stretch=1)

        root_layout.addWidget(sidebar_widget)

        # --- Right side: image grid + nav bar ---
        right_layout = QVBoxLayout()

        self.grid = QGridLayout()
        self.panels: list[ImagePanel] = []
        columns = min(len(self.folders), 2)
        for i, folder in enumerate(self.folders):
            panel = ImagePanel(self._folder_title(i, folder))
            self.panels.append(panel)
            row, col = divmod(i, columns)
            self.grid.addWidget(panel, row, col)
        right_layout.addLayout(self.grid, stretch=1)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("<< Previous")
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button = QPushButton("Next >>")
        self.next_button.clicked.connect(self.show_next)
        self.index_label = QLabel("")
        self.index_label.setAlignment(Qt.AlignCenter)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.index_label, stretch=1)
        nav_layout.addWidget(self.next_button)
        right_layout.addLayout(nav_layout)

        root_layout.addLayout(right_layout, stretch=1)

        self.setStatusBar(QStatusBar())

        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.show_previous)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.show_next)
        QShortcut(QKeySequence(Qt.Key_Home), self, activated=self.show_first)
        QShortcut(QKeySequence(Qt.Key_End), self, activated=self.show_last)

    def _apply_filter(self, text: str) -> None:
        text = text.lower()
        self._filtered = [f for f in self.filenames if text in f.lower()]
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self.list_widget.addItems(self._filtered)
        self.list_widget.blockSignals(False)
        self._sync_list_selection()

    def _on_list_selection(self, row: int) -> None:
        if row < 0 or row >= len(self._filtered):
            return
        self.current_index = self.filenames.index(self._filtered[row])
        self._show_current()

    def show_previous(self) -> None:
        if not self.filenames:
            return
        self.current_index = (self.current_index - 1) % len(self.filenames)
        self._sync_list_selection()
        self._show_current()

    def show_next(self) -> None:
        if not self.filenames:
            return
        self.current_index = (self.current_index + 1) % len(self.filenames)
        self._sync_list_selection()
        self._show_current()

    def show_first(self) -> None:
        if not self.filenames:
            return
        self.current_index = 0
        self._sync_list_selection()
        self._show_current()

    def show_last(self) -> None:
        if not self.filenames:
            return
        self.current_index = len(self.filenames) - 1
        self._sync_list_selection()
        self._show_current()

    def _sync_list_selection(self) -> None:
        if not self.filenames:
            return
        name = self.filenames[self.current_index]
        self.list_widget.blockSignals(True)
        if name in self._filtered:
            self.list_widget.setCurrentRow(self._filtered.index(name))
        else:
            self.list_widget.setCurrentRow(-1)
        self.list_widget.blockSignals(False)

    def _show_current(self) -> None:
        if not self.filenames:
            self.index_label.setText("0 / 0")
            for panel in self.panels:
                panel.set_image(None)
            return

        name = self.filenames[self.current_index]
        for panel, images in zip(self.panels, self.folder_images):
            panel.set_image(images.get(name))

        self.index_label.setText(f"{self.current_index + 1} / {len(self.filenames)}  —  {name}")
        self.statusBar().showMessage(name)


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(args.source_dir, args.processed_dirs)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
