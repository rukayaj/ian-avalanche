from pathlib import Path
from PIL import Image

from src.batch_builder import (
    build_detect_regions_request,
    build_extract_all_request,
    build_extract_wind_request,
    build_extract_precip_request,
    build_extract_temperature_request,
)


def test_build_detect_regions_request_shape():
    # Small dummy image for faster test
    img = Image.new('RGB', (200, 200), (255, 255, 255))
    req = build_detect_regions_request('gpt-5', img)
    assert req['method'] == 'POST'
    assert req['url'] == '/v1/responses'
    body = req['body']
    assert body['model'] == 'gpt-5'
    assert body['response_format']['type'] == 'json_schema'
    assert 'schema' in body['response_format']['json_schema']
    # One text + one image in content
    content = body['input'][0]['content']
    assert content[0]['type'] == 'input_text'
    assert content[1]['type'] == 'input_image'


def test_build_extract_all_request_shape():
    # Create three dummy crops
    crops = {
        'wind': Image.new('RGB', (300, 200), (240, 240, 240)),
        'precipitation': Image.new('RGB', (300, 200), (240, 240, 240)),
        'temperature': Image.new('RGB', (300, 200), (240, 240, 240)),
    }
    req = build_extract_all_request('gpt-5', crops)
    assert req['method'] == 'POST'
    body = req['body']
    assert body['model'] == 'gpt-5'
    assert body['response_format']['json_schema']['name'] == 'combined_series_schema'
    content = body['input'][0]['content']
    # 1 text + 3 images
    assert len(content) == 4
    assert content[0]['type'] == 'input_text'
    assert content[1]['type'] == 'input_image'
    assert content[2]['type'] == 'input_image'
    assert content[3]['type'] == 'input_image'


def test_build_extract_individual_requests_shape():
    crop = Image.new('RGB', (320, 220), (250, 250, 250))
    r1 = build_extract_wind_request('gpt-5', crop)
    assert r1['url'] == '/v1/responses' and r1['body']['response_format']['json_schema']['name'] == 'wind_series_schema'
    r2 = build_extract_precip_request('gpt-5', crop)
    assert r2['body']['response_format']['json_schema']['name'] == 'precip_series_schema'
    r3 = build_extract_temperature_request('gpt-5', crop)
    assert r3['body']['response_format']['json_schema']['name'] == 'temp_series_schema'
