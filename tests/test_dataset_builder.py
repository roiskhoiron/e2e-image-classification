from pathlib import Path
import numpy as np
import cv2

from src.data.dataset_builder import (
    is_image_file,
    validate_image,
    scan_dataset,
    split_dataset,
    log_invalid_files,
)


def create_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    ext = path.suffix.lower()
    if ext == ".png":
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    else:
        cv2.imwrite(str(path), img)


def create_corrupt_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not an image file")


class TestIsImageFile:
    def test_valid_extensions(self):
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            assert is_image_file(Path(f"image{ext}"))

    def test_invalid_extension(self):
        assert not is_image_file(Path("data.txt"))
        assert not is_image_file(Path("script.py"))
        assert not is_image_file(Path("image.gif"))


class TestValidateImage:
    def test_valid_image(self, tmp_path):
        p = tmp_path / "test.png"
        create_dummy_image(p)
        assert validate_image(p)

    def test_corrupt_image(self, tmp_path):
        p = tmp_path / "corrupt.jpg"
        create_corrupt_file(p)
        assert not validate_image(p)


class TestScanDataset:
    def test_scan_valid_images(self, tmp_path):
        for i in range(3):
            create_dummy_image(tmp_path / "class_a" / f"img_{i}.jpg")
        valid, invalid = scan_dataset(tmp_path)
        assert len(valid) == 3
        assert len(invalid) == 0

    def test_scan_mixed(self, tmp_path):
        create_dummy_image(tmp_path / "class_a" / "valid.png")
        create_corrupt_file(tmp_path / "class_a" / "corrupt.jpg")
        create_corrupt_file(tmp_path / "class_a" / "corrupt2.jpg")
        valid, invalid = scan_dataset(tmp_path)
        assert len(valid) == 1
        assert len(invalid) == 2

    def test_scan_non_image_extensions(self, tmp_path):
        create_dummy_image(tmp_path / "class_a" / "img.jpg")
        (tmp_path / "class_a" / "note.txt").write_text("hello")
        valid, invalid = scan_dataset(tmp_path)
        assert len(valid) == 1
        assert len(invalid) == 1


class TestSplitDataset:
    def test_split_creates_directories(self, tmp_path):
        files = []
        for i in range(10):
            p = tmp_path / "raw" / "cats" / f"cat_{i}.jpg"
            create_dummy_image(p)
            files.append(p)
        output = tmp_path / "processed"
        split_dataset(files, output, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        train_files = list((output / "train" / "cats").iterdir())
        val_files = list((output / "val" / "cats").iterdir())
        test_files = list((output / "test" / "cats").iterdir())
        assert len(train_files) == 8
        assert len(val_files) == 1
        assert len(test_files) == 1

    def test_split_two_classes(self, tmp_path):
        files = []
        for i in range(10):
            p = tmp_path / "raw" / "dogs" / f"dog_{i}.jpg"
            create_dummy_image(p)
            files.append(p)
        for i in range(10):
            p = tmp_path / "raw" / "cats" / f"cat_{i}.jpg"
            create_dummy_image(p)
            files.append(p)
        output = tmp_path / "processed"
        split_dataset(files, output, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
        assert len(list((output / "train" / "dogs").iterdir())) == 8
        assert len(list((output / "train" / "cats").iterdir())) == 8


class TestLogInvalidFiles:
    def test_log_creates_file(self, tmp_path):
        log_file = tmp_path / "ingestion_log.txt"
        paths = [tmp_path / "corrupt1.jpg", tmp_path / "corrupt2.png"]
        log_invalid_files(paths, log_file)
        assert log_file.exists()
        content = log_file.read_text()
        assert str(paths[0]) in content
        assert str(paths[1]) in content
