"""Command-line demo for one image or a directory of sample images."""
import argparse
import sys
from pathlib import Path
import cv2
from app.detector import YOLOv4Detector, load_image

def main() -> int:
    parser = argparse.ArgumentParser(description='Pretrained YOLOv4 image detection with OpenCV.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path, help='Path to one image')
    source.add_argument('--input-dir', type=Path, help='Process JPG/JPEG/PNG files in a directory')
    parser.add_argument('--confidence', type=float, default=0.50)
    parser.add_argument('--nms-threshold', type=float, default=0.40)
    parser.add_argument('--input-size', type=int, default=416, help='Square blob size; multiple of 32')
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--no-display', action='store_true', help='Save without opening a window')
    args = parser.parse_args()
    try:
        if args.image:
            paths = [args.image]
        else:
            if not args.input_dir.is_dir():
                raise FileNotFoundError(f'Image directory not found: {args.input_dir}')
            paths = sorted(p for p in args.input_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
            if not paths:
                raise ValueError('No JPG/JPEG/PNG images found in the input directory.')
            if len({p.stem for p in paths}) != len(paths):
                raise ValueError('Input filenames must have distinct stems to avoid output overwrites.')
        first_image = load_image(paths[0])
        detector = YOLOv4Detector(confidence=args.confidence,
                                 nms_threshold=args.nms_threshold, input_size=args.input_size)
        failures = 0
        for index, path in enumerate(paths):
            try:
                image = first_image if index == 0 else load_image(path)
                result = detector.detect(image)
                output = detector.save_output(image, result['detections'],
                                              args.output_dir / f'{path.stem}_detected.jpg')
                print(f'\nImage: {path}\nDetected Objects\n----------------')
                for name, count in result['counts'].items():
                    print(f'{name}: {count}')
                if not result['counts']:
                    print('No objects above the confidence threshold.')
                print(f"Total detections: {result['total_detections']}")
                print(f"Inference time: {result['inference_time_ms']:.2f} ms (local run)")
                print(f'Saved: {output}')
                if not args.no_display:
                    cv2.imshow('YOLOv4 Object Detection', detector.draw_boxes(image, result['detections']))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except (OSError, ValueError, cv2.error) as exc:
                failures += 1
                print(f'Error processing {path}: {exc}', file=sys.stderr)
                if not args.no_display:
                    print('For a system without a display, use --no-display.', file=sys.stderr)
        return 1 if failures else 0
    except (OSError, ValueError, cv2.error) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
