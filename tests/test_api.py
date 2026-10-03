"""API validation checks without real weights or extra test dependencies."""
import asyncio
from io import BytesIO
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from pydantic import TypeAdapter
from starlette.datastructures import Headers
from starlette.requests import Request
from app.api import MAX_UPLOAD_BYTES, UploadResult, app, detect, health, lifespan

class APITests(unittest.TestCase):
    def setUp(self):
        self.request = Request({'type': 'http', 'app': app})
        app.state.detector = None
        _, encoded = cv2.imencode('.png', np.zeros((20, 30, 3), dtype=np.uint8))
        self.png = encoded.tobytes()

    def upload(self, data, filename='image.png', mime='image/png'):
        return UploadFile(file=BytesIO(data), filename=filename,
                          headers=Headers({'content-type': mime}))

    def test_unavailable_model(self):
        with self.assertRaises(HTTPException) as error:
            health(self.request)
        self.assertEqual(error.exception.status_code, 503)
        with self.assertRaises(HTTPException) as error:
            detect(self.request, self.upload(self.png))
        self.assertEqual(error.exception.status_code, 503)

    def test_bad_uploads(self):
        cases = [(b'', 'image.png', 'image/png', 400),
                 (b'not an image', 'image.png', 'image/png', 400),
                 (self.png, 'image.gif', 'image/gif', 415),
                 (self.png, 'image.jpg', 'image/jpeg', 400),
                 (self.png, 'image.png', 'image/jpeg', 400),
                 (b'x' * (MAX_UPLOAD_BYTES + 1), 'image.png', 'image/png', 413)]
        for data, filename, mime, code in cases:
            with self.subTest(code=code, filename=filename):
                file = self.upload(data, filename, mime)
                with self.assertRaises(HTTPException) as error:
                    detect(self.request, file)
                self.assertEqual(error.exception.status_code, code)
                self.assertTrue(file.file.closed)

    def test_lifespan_reuses_detector_and_returns_schema(self):
        result = {'total_detections': 0, 'counts': {}, 'detections': [], 'inference_time_ms': 1.0}
        async def scenario():
            with patch('app.api.YOLOv4Detector') as factory:
                factory.return_value.detect.return_value = result
                async with lifespan(app):
                    self.assertEqual(health(self.request), {'status': 'ok', 'model': 'YOLOv4'})
                    for _ in range(2):
                        output = detect(self.request, self.upload(self.png, 'C:\\fakepath\\image.png'))
                        self.assertEqual(output['filename'], 'image.png')
                        TypeAdapter(UploadResult).validate_python(output)
                    self.assertEqual(factory.call_count, 1)
                    self.assertEqual(factory.return_value.detect.call_count, 2)
                self.assertIsNone(app.state.detector)
        asyncio.run(scenario())

    def test_openapi(self):
        schema = app.openapi()
        self.assertIn('/detect', schema['paths'])
        self.assertIn('multipart/form-data', schema['paths']['/detect']['post']['requestBody']['content'])

if __name__ == '__main__':
    unittest.main()
