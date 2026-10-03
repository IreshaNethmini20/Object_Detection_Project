"""Regression checks using synthetic outputs, not measured model accuracy."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from app.detector import MODEL_DIR, YOLOv4Detector, load_image

class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.detector = YOLOv4Detector.__new__(YOLOv4Detector)
        self.detector.classes = ['person', 'car']
        self.detector.confidence = 0.5
        self.detector.nms_threshold = 0.4

    def test_joint_scores_and_class_wise_nms(self):
        # Same-class duplicate is removed; another class at the same box stays.
        rows = np.array([[.5, .5, .4, .4, .7, .6, .1],
                         [.5, .5, .4, .4, .9, .55, .1],
                         [.5, .5, .4, .4, .9, .1, .8],
                         [.1, .1, .1, .1, .9, .49, .1]], dtype=np.float32)
        detections = self.detector._postprocess([rows], 1000, 500)
        self.assertEqual(len(detections), 2)
        person = next(d for d in detections if d['class_name'] == 'person')
        self.assertAlmostEqual(person['confidence'], .6, places=6)
        self.assertEqual(person['bounding_box'], {'x': 300, 'y': 150, 'width': 400, 'height': 200})

    def test_empty_clipped_and_invalid_boxes(self):
        self.assertEqual(self.detector._postprocess([], 100, 100), [])
        rows = np.array([[0, 0, .4, .4, .9, .8, .1],
                         [.5, .5, -.1, .2, .9, .8, .1],
                         [np.nan, .5, .4, .4, .9, .8, .1]], dtype=np.float32)
        result = self.detector._postprocess([rows], 100, 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bounding_box'], {'x': 0, 'y': 0, 'width': 20, 'height': 20})

    def test_unicode_path_and_annotation_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample_\u732b.png'
            image = np.zeros((80, 120, 3), dtype=np.uint8)
            detection = {'class_id': 0, 'class_name': 'person', 'confidence': .9,
                         'bounding_box': {'x': 10, 'y': 10, 'width': 50, 'height': 40}}
            self.detector.save_output(image, [detection], path)
            self.assertEqual(load_image(path).shape, image.shape)
            self.assertFalse(image.any())
            self.assertTrue(load_image(path).any())
            with self.assertRaises(FileNotFoundError):
                load_image(Path(directory) / 'missing.jpg')
            path.write_bytes(b'not an image')
            with self.assertRaises(ValueError):
                load_image(path)

    def test_model_validation(self):
        for missing in ('yolov4.cfg', 'yolov4.weights', 'coco.names'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                for name in ('yolov4.cfg', 'yolov4.weights', 'coco.names'):
                    if name != missing:
                        (Path(directory) / name).write_text('placeholder')
                with self.assertRaisesRegex(FileNotFoundError, missing):
                    YOLOv4Detector(directory)
        for args in ({'input_size': 415}, {'confidence': 0}, {'nms_threshold': 1.1}):
            with self.assertRaises(ValueError):
                YOLOv4Detector(**args)

    def test_layer_indices_flat_and_nested_load_once(self):
        for indices in (np.array([1, 3]), np.array([[1], [3]])):
            with tempfile.TemporaryDirectory() as directory:
                for name in ('yolov4.cfg', 'yolov4.weights'):
                    (Path(directory) / name).touch()
                (Path(directory) / 'coco.names').write_text('\n'.join(str(i) for i in range(80)))
                with patch('app.detector.cv2.dnn.readNetFromDarknet') as read_net:
                    net = read_net.return_value
                    net.getLayerNames.return_value = ['first', 'middle', 'last']
                    net.getUnconnectedOutLayers.return_value = indices
                    net.forward.return_value = []
                    detector = YOLOv4Detector(directory)
                    image = np.zeros((100, 200, 3), dtype=np.uint8)
                    self.assertEqual(detector.detect(image)['total_detections'], 0)
                    detector.detect(image)
                    self.assertEqual(read_net.call_count, 1)
                    self.assertEqual(detector.output_layers, ['first', 'last'])
                    self.assertEqual(image.shape, (100, 200, 3))

if __name__ == '__main__':
    unittest.main()
