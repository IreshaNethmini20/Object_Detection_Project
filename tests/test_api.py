"""API validation checks without real weights or extra test dependencies."""
import asyncio
import json
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
    """Use synthetic predictions to test validation and HTTP contracts."""
    def setUp(self):
        self.request = Request({'type': 'http', 'app': app})
        app.state.detector = None
        _, encoded = cv2.imencode('.png', np.zeros((20, 30, 3), dtype=np.uint8))
        self.png = encoded.tobytes()

    def upload(self, data, filename='image.png', mime='image/png'):
        """Build an in-memory upload without requiring a web server or weights."""
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

    def http_request(self, path, method='GET', body=b'', headers=()):
        """Exercise the ASGI routes directly without adding an HTTP test library."""
        async def request():
            messages = []
            received = False

            async def receive():
                nonlocal received
                if not received:
                    received = True
                    return {'type': 'http.request', 'body': body, 'more_body': False}
                # File responses may watch for disconnects; remain connected
                # until the response completes rather than ending it early.
                await asyncio.Event().wait()

            async def send(message):
                messages.append(message)

            await app({'type': 'http', 'asgi': {'version': '3.0'},
                       'http_version': '1.1', 'method': method, 'scheme': 'http',
                       'path': path, 'raw_path': path.encode(), 'query_string': b'',
                       'root_path': '', 'headers': list(headers),
                       'server': ('testserver', 80), 'client': ('testclient', 123)}, receive, send)
            start = next(m for m in messages if m['type'] == 'http.response.start')
            content = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
            return start['status'], dict(start['headers']), content
        return asyncio.run(request())

    def test_frontend_and_local_assets(self):
        status, headers, content = self.http_request('/')
        self.assertEqual(status, 200)
        self.assertIn(b'text/html', headers[b'content-type'])
        self.assertIn(b'YOLOv4 Object Detection System', content)
        self.assertIn(b'/static/js/app.js', content)
        for path, marker in (('/static/css/style.css', b'.workspace'),
                             ('/static/js/app.js', b'FormData'),
                             ('/static/favicon.svg', b'<svg')):
            with self.subTest(path=path):
                status, _, content = self.http_request(path)
                self.assertEqual(status, 200)
                self.assertIn(marker, content)
        status, headers, _ = self.http_request('/favicon.ico')
        self.assertEqual(status, 307)
        self.assertEqual(headers[b'location'], b'/static/favicon.svg')

    def test_api_info_swagger_and_health_routes(self):
        status, _, content = self.http_request('/api/info')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(content)['docs'], '/docs')
        self.assertEqual(self.http_request('/docs')[0], 200)
        self.assertEqual(self.http_request('/health')[0], 503)

    def test_http_upload_preserves_response_contract(self):
        # Multipart parsing and response serialization are exercised as well
        # as the route function; synthetic scores are not real model results.
        boundary = b'test-upload-boundary'
        body = (b'--' + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="image.png"'
                b'\r\nContent-Type: image/png\r\n\r\n' + self.png + b'\r\n--' + boundary + b'--\r\n')
        headers = [(b'content-type', b'multipart/form-data; boundary=' + boundary)]
        with patch.object(app.state, 'detector') as detector:
            detector.detect.return_value = {
                'total_detections': 1, 'counts': {'person': 1}, 'inference_time_ms': 1.0,
                'detections': [{'class_id': 0, 'class_name': 'person', 'confidence': .9,
                                'bounding_box': {'x': 1, 'y': 2, 'width': 3, 'height': 4}}]}
            status, _, content = self.http_request('/detect', 'POST', body, headers)
            self.assertEqual(status, 200)
            result = json.loads(content)
            self.assertEqual(result['filename'], 'image.png')
            self.assertEqual(result['counts'], {'person': 1})
            self.assertEqual(result['detections'][0]['bounding_box']['width'], 3)
            TypeAdapter(UploadResult).validate_python(result)
        self.assertEqual(self.http_request('/detect', 'POST', body, headers)[0], 503)

    def test_startup_without_model_leaves_frontend_available(self):
        async def scenario():
            with patch('app.api.YOLOv4Detector', side_effect=FileNotFoundError('Missing weights')):
                with self.assertLogs('app.api', level='WARNING'):
                    async with lifespan(app):
                        self.assertIsNone(app.state.detector)
        asyncio.run(scenario())
        self.assertEqual(self.http_request('/')[0], 200)

if __name__ == '__main__':
    unittest.main()
