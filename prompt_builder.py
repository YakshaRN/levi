"""
Prompt Builder Module for Levitate.
Generates image prompts based on audio features.
"""
import random


# Scene types based on mood and energy combinations
SCENE_OPTIONS = {
    ("melancholic", "minimal"): [
        "abandoned cathedral in rain",
        "lonely lighthouse at dusk",
        "empty train station at night",
        "foggy cemetery with ancient trees"
    ],
    ("melancholic", "low"): [
        "rainy city street at night",
        "solitary figure on misty bridge",
        "autumn forest with falling leaves",
        "old pier during storm"
    ],
    ("emotional", "low"): [
        "sunset over calm ocean",
        "cherry blossoms in moonlight",
        "snow-covered village at dawn",
        "mountain lake at golden hour"
    ],
    ("emotional", "medium"): [
        "aurora borealis over frozen lake",
        "floating lanterns in night sky",
        "ancient temple in cherry blossom garden",
        "waterfall in mystical forest"
    ],
    ("aggressive", "high"): [
        "volcanic eruption with lightning",
        "epic battle on stormy cliffs",
        "dragon emerging from inferno",
        "cyberpunk city in chaos"
    ],
    ("aggressive", "intense"): [
        "apocalyptic cityscape with fire",
        "massive tsunami hitting coast",
        "warship battle in thunderstorm",
        "demon army marching"
    ],
    ("tense", "medium"): [
        "dark alley in noir city",
        "abandoned asylum corridor",
        "foggy forest with glowing eyes",
        "submarine in deep abyss"
    ],
    ("tense", "high"): [
        "storm over gothic castle",
        "chase through neon streets",
        "escape from collapsing temple",
        "standoff in wild west town"
    ],
    ("euphoric", "high"): [
        "fireworks over futuristic city",
        "rave in crystal cave",
        "sunrise from mountain peak",
        "parade through magical kingdom"
    ],
    ("euphoric", "intense"): [
        "supernova explosion in space",
        "festival of lights celebration",
        "phoenix rising from flames",
        "cosmic dance of galaxies"
    ],
    ("bright", "medium"): [
        "sunny meadow with wildflowers",
        "tropical beach paradise",
        "hot air balloons at sunrise",
        "colorful coral reef"
    ],
    ("bright", "high"): [
        "vibrant carnival scene",
        "rainbow over waterfall",
        "kite festival on beach",
        "butterfly garden in bloom"
    ],
    ("driving", "high"): [
        "motorcycle racing through canyon",
        "train through mountain pass",
        "speedboat cutting through waves",
        "car chase in rain"
    ],
    ("driving", "intense"): [
        "rocket launch at night",
        "avalanche in mountains",
        "stampede across savanna",
        "jet flying through canyon"
    ],
    ("epic", "medium"): [
        "ancient castle on cliff",
        "fleet of ships at sunset",
        "army before great battle",
        "throne room of ice palace"
    ],
    ("epic", "high"): [
        "dragon flying over kingdom",
        "giant robot in destroyed city",
        "portal opening between worlds",
        "wizard summoning storm"
    ],
    ("atmospheric", "low"): [
        "misty bamboo forest",
        "underwater ruins",
        "desert oasis at night",
        "cave with bioluminescent plants"
    ],
    ("atmospheric", "medium"): [
        "alien planet landscape",
        "steampunk airship dock",
        "enchanted library",
        "crystal ice palace"
    ],
}

# Color palettes mapped to musical keys (0-11: C to B)
COLOR_PALETTES = {
    0: "deep reds and burgundy",       # C
    1: "coral and rose gold",          # C#
    2: "warm oranges and amber",       # D
    3: "golden yellow and honey",      # D#
    4: "lime green and chartreuse",    # E
    5: "forest green and emerald",     # F
    6: "teal and turquoise",           # F#
    7: "ocean blue and azure",         # G
    8: "indigo and royal blue",        # G#
    9: "deep purple and violet",       # A
    10: "magenta and fuchsia",         # A#
    11: "silver and moonlight white",  # B
}

# Lighting options based on energy level
LIGHTING_OPTIONS = {
    "minimal": "dim ambient glow with single light source",
    "low": "soft diffused lighting with gentle shadows",
    "medium": "balanced cinematic lighting with depth",
    "high": "dramatic volumetric rays with strong contrast",
    "intense": "explosive light bursts with lens flares",
}

# Atmosphere based on texture
ATMOSPHERE_OPTIONS = {
    "smooth": "serene and dreamlike, soft focus edges",
    "textured": "gritty and detailed, visible particles in air",
    "rhythmic": "dynamic motion blur, sense of movement",
    "layered": "rich depth of field, multiple planes of interest",
}

# Style modifiers based on tempo
STYLE_OPTIONS = {
    "slow": "painterly brushstrokes, meditative composition",
    "moderate": "balanced composition, classical framing",
    "upbeat": "dynamic angles, energetic composition",
    "fast": "motion blur, extreme angles, kinetic energy",
}

# Weather/atmosphere options based on mood
WEATHER_OPTIONS = {
    "melancholic": ["gentle rain", "thick fog", "overcast twilight", "autumn mist"],
    "emotional": ["golden hour light", "soft snowfall", "morning dew", "starlit night"],
    "aggressive": ["raging thunderstorm", "volcanic smoke", "dust storm", "fire embers"],
    "tense": ["approaching storm", "eerie fog", "eclipse darkness", "ominous clouds"],
    "euphoric": ["brilliant sunshine", "rainbow after rain", "northern lights", "meteor shower"],
    "bright": ["clear blue sky", "puffy white clouds", "spring sunshine", "crystal clear air"],
    "driving": ["wind and dust", "rain streaks", "snow flurry", "heat haze"],
    "epic": ["dramatic clouds", "god rays", "magical aurora", "cosmic nebula"],
    "atmospheric": ["mystical haze", "ethereal glow", "ambient fog", "dappled light"],
}

# Default fallback scenes
DEFAULT_SCENES = [
    "mystical landscape at twilight",
    "dramatic natural vista",
    "fantastical environment"
]


def build_prompt(features: dict) -> str:
    """
    Build an image generation prompt based on audio features.
    
    Args:
        features: Dictionary of audio features from analyze_audio()
        
    Returns:
        A detailed prompt string for image generation
    """
    tempo = features["tempo"]
    tempo_class = features["tempo_class"]
    energy = features["energy"]
    mood = features["mood"]
    texture = features["texture"]
    dominant_pitch = features["dominant_pitch"]
    
    # Select scene
    scene = _select_scene(mood, energy)
    
    # Build color style
    color_style = _build_color_style(dominant_pitch, mood)
    
    # Get lighting
    lighting = LIGHTING_OPTIONS.get(energy, "cinematic lighting")
    
    # Get atmosphere
    atmosphere = ATMOSPHERE_OPTIONS.get(texture, "atmospheric perspective")
    
    # Get style
    style = STYLE_OPTIONS.get(tempo_class, "cinematic composition")
    
    # Get weather
    weather = _select_weather(mood)

    prompt = f"""{scene}, {weather}.
Color palette: {color_style}.
Lighting: {lighting}.
Atmosphere: {atmosphere}.
Style: {style}.
Ultra detailed digital art, 8K resolution, trending on ArtStation.
Cinematic wide shot, professional concept art, no text, no watermark, no signature."""

    return prompt.strip()


def _select_scene(mood: str, energy: str) -> str:
    """Select a scene based on mood and energy."""
    key = (mood, energy)
    
    # Try exact match first
    if key in SCENE_OPTIONS:
        return random.choice(SCENE_OPTIONS[key])
    
    # Try to find a close match
    for m in [mood, "atmospheric"]:
        for e in [energy, "medium"]:
            if (m, e) in SCENE_OPTIONS:
                return random.choice(SCENE_OPTIONS[(m, e)])
    
    return random.choice(DEFAULT_SCENES)


def _build_color_style(dominant_pitch: int, mood: str) -> str:
    """Build color style based on musical key and mood."""
    base_palette = COLOR_PALETTES.get(dominant_pitch, "rich jewel tones")
    
    if mood in ["melancholic", "tense"]:
        return f"muted {base_palette} with deep shadows"
    elif mood in ["aggressive", "driving"]:
        return f"intense {base_palette} with stark contrasts"
    elif mood in ["euphoric", "bright"]:
        return f"vibrant {base_palette} with luminous highlights"
    else:
        return f"{base_palette} with cinematic grading"


def _select_weather(mood: str) -> str:
    """Select weather/atmosphere based on mood."""
    options = WEATHER_OPTIONS.get(mood, ["dramatic sky"])
    return random.choice(options)
