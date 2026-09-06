from fairflex.scenarios import ReplayWindow
from scripts.v3.run_daywise_matrix import _day_windows


def test_day_windows_partition_a_full_week_without_crossing_its_bounds():
    source = ReplayWindow.from_config(["2019-10-01T00:00:00Z", "2019-10-08T00:00:00Z"])

    windows = _day_windows(source)

    assert len(windows) == 7
    assert windows[0].start == source.start
    assert windows[-1].end == source.end
    assert all(first.end == second.start for first, second in zip(windows, windows[1:]))
