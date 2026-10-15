"""Named starting points: canvas sizes and lyrics looks (same as the YABBE editor)."""

CANVAS = {
    "youtube": {"width": 1920, "height": 1080, "fps": 30},
    "youtube_4k": {"width": 3840, "height": 2160, "fps": 30},
    "720p": {"width": 1280, "height": 720, "fps": 30},
    "vertical": {"width": 1080, "height": 1920, "fps": 30},  # Reels / TikTok / Shorts
    "square": {"width": 1080, "height": 1080, "fps": 30},
    "portrait_4_5": {"width": 1080, "height": 1350, "fps": 30},  # Instagram feed
    "cinema": {"width": 1920, "height": 816, "fps": 24},  # 2.35:1
}

LYRICS = {
    "karaoke": ("Words fill in as they're sung", {}),
    "highlight": ("A box jumps to the word being sung", {
        "word_fx": "highlight", "box": "active", "box_color": "#ffffff", "active_box_color": "#ffffff",
        "active_color": "#111111", "upcoming_color": "#ffffff", "past_color": "#ffffff", "outline": 0, "shadow": 2,
        "box_padding": 8}),
    "typewriter": ("Words appear one by one", {
        "word_fx": "reveal", "font": "DejaVu Sans Mono", "bold": False, "case": "lower", "line_in": "none",
        "line_out": "fade", "outline": 2}),
    "pop": ("Two big words at a time, popping as they're sung", {
        "layout": "words", "words_shown": 2, "word_fx": "pop", "size": 130, "case": "upper", "anchor": "middle",
        "active_color": "#ffd43b", "upcoming_color": "#ffffff", "line_in": "zoom", "transition_ms": 150, "outline": 5}),
    "bounce": ("Each word jumps when it's sung", {
        "word_fx": "bounce", "active_color": "#69db7c", "active_scale": 1.15, "upcoming_color": "#ffffffb3",
        "line_in": "slide_up"}),
    "big": ("One huge word at a time", {
        "layout": "words", "words_shown": 1, "word_fx": "none", "size": 190, "case": "upper", "anchor": "middle",
        "line_in": "zoom", "line_out": "none", "transition_ms": 120, "outline": 6, "shadow": 4}),
    "scroll": ("The song scrolls by, the current line in the middle", {
        "layout": "scroll", "lines_shown": 5, "word_fx": "highlight", "anchor": "middle", "other_lines_opacity": 0.5,
        "size": 60, "upcoming_color": "#ffffff", "active_color": "#ffd43b", "past_color": "#ffffff",
        "transition_ms": 300}),
    "stack": ("The current line with the next one under it", {
        "layout": "stack", "lines_shown": 2, "word_fx": "fill", "other_lines_opacity": 0.5, "line_in": "slide_up"}),
    "verse": ("The whole verse, left aligned, words fading in", {
        "layout": "page", "lines_shown": 4, "word_fx": "fade", "align": "left", "anchor": "middle", "size": 64,
        "upcoming_color": "#ffffff59", "past_color": "#ffffff", "other_lines_opacity": 1}),
    "neon": ("The sung word glows", {
        "word_fx": "glow", "active_color": "#ff2bd6", "color": "#ffffff", "outline": 1.5, "outline_color": "#ff2bd6",
        "shadow": 0, "upcoming_color": "#ffffff99", "past_color": "#ffffff"}),
    "boxed": ("Each word on its own label", {
        "word_fx": "highlight", "box": "words", "box_color": "#000000cc", "outline": 0, "shadow": 0,
        "active_color": "#ffd43b", "upcoming_color": "#ffffff", "past_color": "#ffffff", "box_padding": 10}),
    "subtitle": ("Plain lines on a dark strip", {
        "word_fx": "none", "size": 56, "bold": False, "box": "line", "box_color": "#000000aa", "outline": 0,
        "shadow": 0, "margin": 0.06, "box_padding": 14}),
}


def lyrics_style(preset: str, overrides: dict | None = None, keep: dict | None = None) -> dict:
    if preset not in LYRICS:
        raise ValueError(f"Unknown lyrics preset {preset!r}; pick one of: {', '.join(LYRICS)}")
    timing = {k: keep[k] for k in ("offset_ms", "lead_ms", "hold_ms") if keep and k in keep}
    return {**LYRICS[preset][1], **timing, **(overrides or {}), "preset": preset}
