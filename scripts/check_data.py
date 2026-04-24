from pathlib import Path


def main(root: str = "data") -> None:
    root = Path(root)
    for split in ["train", "val", "test"]:
        split_path = root / split
        if not split_path.exists():
            print(f"[缺失] {split_path}")
            continue

        classes = [p.name for p in split_path.iterdir() if p.is_dir()]
        print(f"[{split}] 类别数: {len(classes)}")
        for cls in sorted(classes):
            img_count = len(list((split_path / cls).glob("*")))
            print(f"  - {cls}: {img_count}")


if __name__ == "__main__":
    main()
