from pathlib import Path
import shutil
import exifread
import argparse

RAW_EXTENSIONS = {".raf", ".dng", ".nef", ".cr2", ".cr3", ".arw", ".orf", ".rw2"}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--min-iso", type=int, default=0)
parser.add_argument("--max-iso", type=int)
args = parser.parse_args()

out_dir = Path(args.output)

out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(p for p in Path(args.input).rglob("*") if p.suffix.lower() in RAW_EXTENSIONS)

for file in files:
    with open(file, "rb") as f:
        print(f"{file.name}")

        try:
            tags = exifread.process_file(f)

            iso = tags.get("EXIF ISOSpeedRatings") or tags.get("Image ISO")
            # shutter_speed = tags.get("EXIF ExposureTime")

            try:
                # shutter_speed = eval(str(shutter_speed))
                iso = int(str(iso))
            except ValueError as e:
                iso = 9999999
                # shutter_speed = 0
                print(f"  unable to determine iso. {e}")

            if iso >= args.min_iso and iso <= args.max_iso:  # and shutter_speed <= 1/100:
                shutil.copy(file, out_dir)
                print("  accepted")
            else:
                print("  rejected")
        except Exception as e:
            print(f"  error occured {e}")
