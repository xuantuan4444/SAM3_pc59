"""
check_pc59_class_order.py
==========================
Diagnostic: compares the class order your current sam3_base_pc59_nollm.py derives from
59_labels.txt against the alphabetical order actually used to build SegmentationClassContext/
(prepare_pc59_mat_to_png.py's LUT). Prints every index where they disagree -- those are exactly
the classes that will silently get near-0 IoU even when SAM3's mask is visually correct, because
GT and prediction are comparing different classes at the same index.

Usage: put this in the same folder as your sam3_base_pc59_nollm.py (so it can reuse
59_contexts/59_labels.txt), then run: python check_pc59_class_order.py
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LABELS_TXT_PATH = PROJECT_ROOT / "59_contexts" / "59_labels.txt"  # TODO: adjust if needed


def parse_pc59_labels(path):
    id_to_name = {}
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    for line_no, line in enumerate(lines, start=1):
        idx, name = None, None
        if ":" in line:
            left, right = line.split(":", 1)
            if left.strip().isdigit():
                idx, name = int(left.strip()), right.strip()
        if idx is None:
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                idx, name = int(parts[0]), parts[1].strip()
        if idx is None:
            idx, name = line_no, line
        id_to_name[idx] = name
    return id_to_name


# --- Order A: exactly what sam3_base_pc59_nollm.py currently does ---
id_to_name = parse_pc59_labels(LABELS_TXT_PATH)
order_from_labels_txt = [id_to_name[i] for i in sorted(id_to_name)]

# --- Order B: the CORRECT order, matching prepare_pc59_mat_to_png.py's LUT and every other
# PC59 file in the pipeline (verified against mmsegmentation's PascalContextDataset59.CLASSES) ---
correct_alphabetical_order = [
    "aeroplane", "bag", "bed", "bedclothes", "bench", "bicycle", "bird", "boat", "book",
    "bottle", "building", "bus", "cabinet", "car", "cat", "ceiling", "chair", "cloth",
    "computer", "cow", "cup", "curtain", "dog", "door", "fence", "floor", "flower", "food",
    "grass", "ground", "horse", "keyboard", "light", "motorbike", "mountain", "mouse",
    "person", "plate", "platform", "pottedplant", "road", "rock", "sheep", "shelves",
    "sidewalk", "sign", "sky", "snow", "sofa", "table", "track", "train", "tree", "truck",
    "tvmonitor", "wall", "water", "window", "wood",
]

print(f"59_labels.txt parsed order : {len(order_from_labels_txt)} classes")
print(f"Correct alphabetical order  : {len(correct_alphabetical_order)} classes")
print()

if order_from_labels_txt == correct_alphabetical_order:
    print("MATCH -- the two orders are identical. The class-order theory is NOT your bug;")
    print("something else is going on (see the other checks below in my reply).")
else:
    n_mismatch = sum(1 for a, b in zip(order_from_labels_txt, correct_alphabetical_order) if a != b)
    print(f"MISMATCH at {n_mismatch}/59 positions. Detail (index: 59_labels.txt -> correct):")
    print(f"{'idx':>4}  {'from 59_labels.txt':<20} {'correct (alphabetical)':<20} {'match?'}")
    for i, (a, b) in enumerate(zip(order_from_labels_txt, correct_alphabetical_order)):
        mark = "" if a == b else "  <-- MISMATCH"
        print(f"{i:>4}  {a:<20} {b:<20}{mark}")
