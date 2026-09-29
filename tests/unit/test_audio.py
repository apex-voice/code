import numpy as np

from apex_voice.harness.audio import (
    PlacedClip,
    assemble_stereo,
    cut_to_ms,
    intersect_total,
    isolated_channel,
    merge_intervals,
    resample,
    timeline_stats,
)


def test_resample_length():
    a = np.ones(16000, dtype=np.float32)
    r = resample(a, 16000, 24000)
    assert abs(len(r) - 24000) <= 1


def test_stereo_channel_separation():
    u = np.ones(1600, dtype=np.float32)  # 100ms @16k
    ag = 0.5 * np.ones(1600, dtype=np.float32)
    clips = [PlacedClip("user", 0, u), PlacedClip("agent", 0, ag)]
    stereo = assemble_stereo(clips, 16000, tail_ms=0)
    # left = user (1.0), right = agent (0.5)
    assert np.allclose(stereo[:1600, 0], 1.0)
    assert np.allclose(stereo[:1600, 1], 0.5)
    assert np.allclose(isolated_channel(clips, "user", 16000, 0)[:1600], 1.0)


def test_interval_math():
    assert merge_intervals([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert intersect_total([(0, 10)], [(5, 15)]) == 5.0


def test_cut_to_ms():
    a = np.ones(1600, dtype=np.float32)
    cut = cut_to_ms(a, 50, 16000, fade_ms=10)  # keep 50ms of 100ms
    assert len(cut) == 800
    assert cut[-1] < 0.5  # faded out


def test_timeline_stats_overlap():
    u = np.ones(1600, dtype=np.float32)  # user 0-100ms
    ag = np.ones(1600, dtype=np.float32)  # agent 50-150ms
    clips = [PlacedClip("user", 0, u), PlacedClip("agent", 50, ag)]
    st = timeline_stats(clips, 16000, tail_ms=0)
    assert abs(st.overlap_ms - 50) < 2
