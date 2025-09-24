import colorsys
import random
import json

def generate_palette_hex(style="analogous", base_color=(0.5, 0.5, 0.5)):
    """Generate a single 4-color palette in HEX format."""
    r, g, b = base_color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    palette = []

    if style == "complementary":
        hues = [h, (h + 0.5) % 1.0, (h + 0.1) % 1.0, (h - 0.1) % 1.0]
    elif style == "analogous":
        step = 0.1
        hues = [(h + (i - 1.5) * step) % 1.0 for i in range(4)]
    elif style == "triadic":
        hues = [h, (h + 1/3.0) % 1.0, (h + 2/3.0) % 1.0, (h + 1/6.0) % 1.0]
    elif style == "monochromatic":
        return [
            rgb_to_hex(colorsys.hls_to_rgb(h, max(0, min(l + delta, 1)), s))
            for delta in [-0.2, -0.1, 0.1, 0.2]
        ]
    elif style == "pastel":
        return [rgb_to_hex(colorsys.hls_to_rgb(random.random(), 0.8, 0.4)) for _ in range(4)]
    elif style == "vibrant":
        return [rgb_to_hex(colorsys.hls_to_rgb(random.random(), 0.5, 1.0)) for _ in range(4)]
    else:
        hues = [h, h, h, h]

    return [rgb_to_hex(colorsys.hls_to_rgb(hue, l, s)) for hue in hues]

def rgb_to_hex(rgb_tuple):
    """Convert RGB tuple (0–1) to HEX string."""
    return '#%02x%02x%02x' % tuple(int(c*255) for c in rgb_tuple)

def generate_multiple_palettes(n_palettes=50000, style="analogous"):
    """Generate multiple palettes in JSON format."""
    palettes = []
    for _ in range(n_palettes):
        base_color = (random.random(), random.random(), random.random())
        palette = generate_palette_hex(style=style, base_color=base_color)
        palettes.append({"colors": palette})
    return palettes

# Generate 50k palettes
palettes = generate_multiple_palettes(n_palettes=50000, style="analogous")

# Save to JSON file
with open("data/palettes.json", "w") as f:
    json.dump(palettes, f, indent=2)

print("Generated 50,000 palettes saved to palettes.json")
