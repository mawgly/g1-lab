"""The site's animated WebP clips as H.264 MP4 (humanoid_sim.site_video) and their place in the static export."""
import json

import pytest
from PIL import Image

from humanoid_sim import i18n, journal, site_export, site_video

WEB = i18n.WEB


def animated_webp(path, size=(33, 21), frames=6, duration=40):
    """A small animated WebP with alpha, as humanoid_sim.clip writes them (odd sizes on purpose)."""
    images = [Image.new('RGBA', size, (40 * index, 120, 200, 255)) for index in range(frames)]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0, lossless=True)
    return path


def test_a_webp_becomes_an_h264_main_yuv420p_faststart_mp4(tmp_path):
    source = animated_webp(tmp_path / 'clip.webp')
    assert site_video.animated(source) and site_video.frame_durations(source) == [40] * 6
    seconds = site_video.encode(source, tmp_path / 'clip.mp4')
    assert seconds == pytest.approx(0.24)
    info = site_video.probe(tmp_path / 'clip.mp4')
    assert 'h264 (Main)' in info['video'] and 'yuv420p' in info['video'] and '34x22' in info['video']
    assert info['duration'] == pytest.approx(0.24, abs=0.05)
    data = (tmp_path / 'clip.mp4').read_bytes()
    assert info['faststart'] and 0 < data.index(b'moov') < data.index(b'mdat')
    assert b'soun' not in data  # no sound track


def test_uneven_frame_durations_keep_their_timing():
    rate, repeats = site_video.timeline([40, 40, 120, 40])
    assert rate == 25 and repeats == [1, 1, 3, 1]
    rate, repeats = site_video.timeline([30, 45, 45])
    assert float(rate) == pytest.approx(1000 / 30) and sum(repeats) == 4


def test_a_still_webp_is_not_a_clip(tmp_path):
    Image.new('RGB', (8, 8)).save(tmp_path / 'still.webp')
    (tmp_path / 'broken.webp').write_bytes(b'RIFF clip')
    assert not site_video.animated(tmp_path / 'still.webp') and not site_video.animated(tmp_path / 'broken.webp')


def test_the_cache_encodes_a_clip_once(tmp_path, monkeypatch):
    source = animated_webp(tmp_path / 'clip.webp')
    first = site_video.convert(source, tmp_path / 'cache')
    assert first.parent == tmp_path / 'cache' and site_video.md5(source) in first.name
    monkeypatch.setattr(site_video, 'encode', lambda *args: pytest.fail('encoded again'))
    assert site_video.convert(source, tmp_path / 'cache') == first
    results = site_video.convert_all([source, source], tmp_path / 'cache')
    assert results == {source: first}


def test_copies_of_one_clip_are_encoded_once(tmp_path, monkeypatch):
    sources = [animated_webp(tmp_path / f'copy{k}.webp') for k in range(4)]
    calls = []
    encode = site_video.encode
    monkeypatch.setattr(site_video, 'encode', lambda source, target: calls.append(source) or encode(source, target))
    results = site_video.convert_all(sources, tmp_path / 'cache')
    assert len(calls) == 1 and len(set(results.values())) == 1 and all(path.is_file() for path in results.values())


def test_the_export_writes_mp4_instead_of_webp_and_renames_the_links(tmp_path, monkeypatch):
    from test_site_export import lab_with_journal  # tests/ is on sys.path (pytest's prepend import mode)

    lab = lab_with_journal(tmp_path)
    folder = journal.folder(lab.root)
    clip = animated_webp(tmp_path / 'walk.webp')
    Image.new('RGB', (33, 21)).save(tmp_path / 'walk-poster.jpg')
    journal.add_entry(lab.root, 'second', 'Идёт', 'Робот идёт.', clip=clip, at=2)
    animated_webp(folder / 'site-ru-walk.webp', frames=5)
    shown = [name for page in site_export.site_pages()
             for name in site_export.PAGE_MEDIA.findall((WEB / page).read_text(encoding='utf-8'))]
    page_clips = [name for name in shown if name.endswith('.webp') and 'poster' not in name]
    for k, name in enumerate(page_clips):
        animated_webp(folder / name, frames=7 + k)  # each its own content (the cache keys by md5)
    # One clip fails to convert: it goes out as its WebP and is reported.
    encode = site_video.encode
    monkeypatch.setattr(site_video, 'encode', lambda source, target: (
        (_ for _ in ()).throw(RuntimeError('ffmpeg 1: broken')) if 'site-ru-walk' in str(source)
        else encode(source, target)))

    out = tmp_path / 'site'
    summary = site_export.export(out, lab.original, video_cache=tmp_path / 'cache')
    media = {path.name for path in (out / 'media').iterdir()}
    assert {'second-clip.mp4', 'second-clip.jpg'} <= media and 'second-clip.webp' not in media
    assert 'first-clip.webp' in media  # not a real WebP: copied as it is
    assert {'site-ru-walk.webp', 'site-ru-walk-poster.jpg'} <= media and 'site-ru-walk.mp4' not in media
    assert [name for name in summary['missing'] if name.startswith(('media/', 'mp4/'))] == ['mp4/site-ru-walk.webp'], \
        summary['video']['failed']
    assert 'broken' in summary['video']['failed']['site-ru-walk.webp']
    assert summary['video']['mp4'] == 1 + len(page_clips)
    for lang in i18n.LANGUAGES:
        data = json.loads((out / 'data' / f'journal.{lang}.json').read_text(encoding='utf-8'))
        second = next(entry for entry in data['entries'] if entry['id'] == 'second')
        assert (second['clip'], second['poster']) == ('second-clip.mp4', 'second-clip.jpg')
        assert '.webp"' not in json.dumps(second)
    for name in page_clips:
        assert name[:-5] + '.mp4' in media and name not in media
    if (out / 'world.html').is_file():
        world = (out / 'world.html').read_text(encoding='utf-8')
        assert page_clips and all(f'data-media="{name[:-5]}.mp4"' in world for name in page_clips)
        assert 'data-media="' in world and not any(f'data-media="{name}"' in world for name in page_clips)
        posters = site_export.PAGE_MEDIA.findall((WEB / 'world.html').read_text(encoding='utf-8'))
        assert all(f'data-poster="{name}"' in world for name in posters if 'poster' in name or name.endswith('.jpg'))
    assert 'video/mp4' in (out / '.htaccess').read_text(encoding='utf-8')
