"""Build a GitHub-flavored reference.docx for pandoc."""

from docx import Document
from docx.shared import Pt, RGBColor

BODY_FONT = "Segoe UI"
CODE_FONT = "Consolas"
H_COLOR   = RGBColor(0x1f, 0x23, 0x28)

HEADING_SIZES = {
    "Heading 1": (26, True),
    "Heading 2": (20, True),
    "Heading 3": (16, True),
    "Heading 4": (14, True),
    "Heading 5": (13, True),
    "Heading 6": (12, False),
}

doc = Document("D:/math-for-ai/ref_default.docx")

for style in doc.styles:
    name = style.name
    if name in HEADING_SIZES:
        size, bold = HEADING_SIZES[name]
        style.font.name      = BODY_FONT
        style.font.size      = Pt(size)
        style.font.bold      = bold
        style.font.color.rgb = H_COLOR
        style.paragraph_format.space_before = Pt(16)
        style.paragraph_format.space_after  = Pt(4)
    elif name == "Normal":
        style.font.name = BODY_FONT
        style.font.size = Pt(11)
        style.paragraph_format.space_after  = Pt(8)
        style.paragraph_format.line_spacing = Pt(18)
    elif "Code" in name or name in ("Verbatim Char", "verbatim"):
        style.font.name = CODE_FONT
        style.font.size = Pt(10)

doc.save("D:/math-for-ai/reference/reference.docx")
print("Done")
