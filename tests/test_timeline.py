"""Timeline edits on plain project documents (no YABBE needed)."""

import pytest

from yabbe_mcp.timeline import Edit, EditError, end


def project():
    return {
        "id": "p", "name": "t", "main_sequence_id": "s",
        "assets": [
            {"id": "v", "kind": "video", "original_name": "beach.mp4", "duration": 20, "width": 1920, "height": 1080,
             "has_audio": True, "status": "ready"},
            {"id": "m", "kind": "audio", "original_name": "song.mp3", "duration": 120, "has_audio": True, "status": "ready"},
            {"id": "i", "kind": "image", "original_name": "logo.png", "duration": 0, "width": 500, "height": 500,
             "status": "ready"},
        ],
        "sequences": [{
            "id": "s", "name": "Main", "settings": {"width": 1920, "height": 1080, "fps": 30, "background": "#000000"},
            "tracks": [{"id": "v2", "kind": "video", "name": "Video 2"}, {"id": "v1", "kind": "video", "name": "Video 1"},
                       {"id": "a1", "kind": "audio", "name": "Audio 1"}],
            "clips": [],
        }],
    }


def test_add_appends_on_free_tracks():
    e = Edit(project())
    a = e.add_media("v", in_point=2, duration=5)
    assert (a["track_id"], a["start"], a["duration"], a["in_point"]) == ("v1", 0, 5, 2)  # lowest video track
    b = e.add_media("v", in_point=10)  # appended, to the end of the file
    assert (b["start"], b["duration"]) == (5, 10)
    m = e.add_media("m")
    assert m["track_id"] == "a1" and m["start"] == 0  # audio appends after audio
    logo = e.add_media("i", start=1, duration=3)  # v1 is busy at 1s: goes on Video 2
    assert logo["track_id"] == "v2" and logo["duration"] == 3
    with pytest.raises(EditError):
        e.add_media("v", start=0, track="Video 1")  # occupied
    with pytest.raises(EditError):
        e.add_media("m", track="Video 1")  # wrong kind
    t = e.track("1")
    assert t["id"] == "v2"


def test_split_trim_speed_keep_footage_and_keys():
    e = Edit(project())
    c = e.add_media("v", in_point=4, duration=10)
    c["keyframes"] = {"opacity": [{"t": 0, "v": 0}, {"t": 6, "v": 1}]}
    c["markers"] = [{"id": "x", "t": 7, "label": "hit"}]
    [(left, right)] = e.split(c["id"], 5)
    r = e.clip(right)
    assert e.clip(left)["duration"] == 5 and r["start"] == 5 and r["in_point"] == 9
    assert r["keyframes"]["opacity"][1]["t"] == 1 and r["markers"][0]["t"] == 2
    # trim the right half's start later: footage stays in place
    e.trim(right, new_start=6)
    assert (r["start"], r["in_point"], r["duration"]) == (6, 10, 4)
    # can't extend past the source (ends at 14 + ...)
    e.trim(right, new_end=100)
    assert end(r) == pytest.approx(6 + (20 - 10))
    e.set_speed(right, 2)
    assert r["duration"] == pytest.approx(5) and r["speed"] == 2 and "speed" in r["effects"]
    assert r["markers"][0]["t"] == pytest.approx(0.5)


def test_delete_ripple_and_linked_moves():
    e = Edit(project())
    a = e.add_media("v", duration=4)
    b = e.add_media("v", duration=4)
    c = e.add_media("v", duration=4)
    e.delete([b["id"]], ripple=True)
    assert c["start"] == 4
    audio = e.separate_audio(a["id"])
    assert audio["link"] == a["link"] and a["audio_detached"]
    e.move(a["id"], start=0)  # no-op move ok
    with pytest.raises(EditError):
        e.move(a["id"], start=2)  # would overlap c on Video 1
    e.move(c["id"], start=20)
    e.move(a["id"], start=10)
    assert audio["start"] == 10  # linked audio moved along


def test_transition_needs_touching_clip():
    e = Edit(project())
    a = e.add_media("v", duration=4)
    with pytest.raises(EditError):
        e.set_transition(a["id"], "mix")
    e.add_media("v", duration=4)
    tr = e.set_transition(a["id"], "wipe_left", 10)
    assert tr["duration"] == 4 and tr["kind"] == "wipe_left"  # capped to the shorter clip


def test_freeze_and_nest():
    e = Edit(project())
    a = e.add_media("v", duration=10)
    after = e.add_media("v", duration=2)
    still = e.freeze(a["id"], 4, "img", 2)
    assert still["type"] == "image" and still["start"] == 4 and still["duration"] == 2
    assert a["duration"] == 4 and after["start"] == 12
    right = next(c for c in e.clips if c["type"] == "video" and c["start"] == 6)
    assert right["in_point"] == 4
    seq, nested = e.nest([a["id"], still["id"]], "Intro")
    assert nested["type"] == "sequence" and nested["duration"] == 6 and seq["name"] == "Intro"
    assert len(seq["clips"]) == 2 and not any(c["id"] == a["id"] for c in e.clips)
    with pytest.raises(EditError):
        e.delete_sequence(seq["id"])  # it's used


def test_describe_is_compact():
    e = Edit(project())
    c = e.add_media("v", duration=3)
    e.patch(c["id"], {"transform": {"scale": 1.5}, "volume": 0.5})
    d = e.describe()
    clip = d["tracks_top_to_bottom"][1]["clips"][0]
    assert clip["name"] == "beach.mp4" and clip["transform"] == {"scale": 1.5} and clip["volume"] == 0.5
    with pytest.raises(EditError):
        e.patch(c["id"], {"start": 3})
