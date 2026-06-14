"""
Build a PDF from the three peer review markdown files using reportlab.
Handles unicode characters gracefully.
"""
import re
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak,
    Table, TableStyle, ListFlowable, ListItem
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register DejaVu fonts for unicode support
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
try:
    pdfmetrics.registerFont(TTFont('DejaVuSerif', f'{FONT_DIR}/DejaVuSerif.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuSerif-Bold', f'{FONT_DIR}/DejaVuSerif-Bold.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuSerif-Italic', f'{FONT_DIR}/DejaVuSerif-Italic.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuSerif-BoldItalic', f'{FONT_DIR}/DejaVuSerif-BoldItalic.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVuSansMono', f'{FONT_DIR}/DejaVuSansMono.ttf'))
    BASE_FONT = 'DejaVuSerif'
    BOLD_FONT = 'DejaVuSerif-Bold'
    ITALIC_FONT = 'DejaVuSerif-Italic'
    MONO_FONT = 'DejaVuSansMono'
    print("DejaVu fonts loaded successfully")
except Exception as e:
    print(f"DejaVu fonts not available: {e}, falling back to Helvetica")
    BASE_FONT = 'Helvetica'
    BOLD_FONT = 'Helvetica-Bold'
    ITALIC_FONT = 'Helvetica-Oblique'
    MONO_FONT = 'Courier'

BASE_DIR = "/home/user/workspace/aprr-multi-agent-routing/paper"
FILES = ["review_harvard.md", "review_mit.md", "author_response.md"]
OUTPUT = os.path.join(BASE_DIR, "peer_review_package.pdf")

TITLE_COLORS = {
    "review_harvard.md": colors.HexColor("#1a237e"),   # Harvard crimson-navy
    "review_mit.md": colors.HexColor("#8b0000"),        # MIT crimson
    "author_response.md": colors.HexColor("#1b5e20"),   # Green for authors
}

def escape_xml(text):
    """Escape XML special chars for ReportLab paragraphs."""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text

def inline_markup(text):
    """Convert simple markdown inline markup to ReportLab XML."""
    # Bold-italic: ***text***
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
    # Bold: **text**
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic: *text* or _text_
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Code: `text`
    text = re.sub(r'`([^`]+)`', lambda m: f'<font name="{MONO_FONT}" size="9">{m.group(1)}</font>', text)
    return text

def parse_markdown_to_story(md_text, accent_color, styles):
    """Parse markdown text into a list of ReportLab flowables."""
    story = []
    lines = md_text.split('\n')
    i = 0
    in_table = False
    table_data = []
    in_code_block = False
    code_lines = []

    # Styles
    h1_style = ParagraphStyle('H1', fontName=BOLD_FONT, fontSize=16,
                               textColor=accent_color, spaceAfter=14, spaceBefore=20,
                               leading=20)
    h2_style = ParagraphStyle('H2', fontName=BOLD_FONT, fontSize=13,
                               textColor=accent_color, spaceAfter=8, spaceBefore=14,
                               leading=16)
    h3_style = ParagraphStyle('H3', fontName=BOLD_FONT, fontSize=11,
                               textColor=colors.HexColor("#333333"), spaceAfter=6, spaceBefore=10,
                               leading=14)
    body_style = ParagraphStyle('Body', fontName=BASE_FONT, fontSize=10,
                                spaceAfter=6, spaceBefore=2, leading=14,
                                alignment=TA_JUSTIFY)
    bullet_style = ParagraphStyle('Bullet', fontName=BASE_FONT, fontSize=10,
                                   spaceAfter=4, spaceBefore=2, leading=13,
                                   leftIndent=18, firstLineIndent=-10,
                                   alignment=TA_LEFT)
    code_style = ParagraphStyle('Code', fontName=MONO_FONT, fontSize=8,
                                 spaceAfter=6, spaceBefore=4, leading=11,
                                 leftIndent=18, backColor=colors.HexColor("#f5f5f5"))

    def flush_table():
        nonlocal table_data, in_table
        if not table_data:
            return []
        flowables = []
        # Filter out separator rows (all dashes)
        cleaned = []
        for row in table_data:
            if all(re.match(r'^[-: ]+$', cell.strip()) for cell in row):
                continue
            cleaned.append(row)
        if not cleaned:
            return []

        # Normalise column count
        max_cols = max(len(r) for r in cleaned)
        for r in cleaned:
            while len(r) < max_cols:
                r.append('')

        # Build table with styled cells
        pdf_rows = []
        for ri, row in enumerate(cleaned):
            pdf_row = []
            for ci, cell in enumerate(row):
                cell = cell.strip()
                cell = escape_xml(cell)
                cell = inline_markup(cell)
                if ri == 0:
                    p = Paragraph(f'<b>{cell}</b>',
                                  ParagraphStyle('TH', fontName=BOLD_FONT, fontSize=9,
                                                  textColor=colors.white, leading=12))
                else:
                    p = Paragraph(cell,
                                  ParagraphStyle('TD', fontName=BASE_FONT, fontSize=9, leading=12))
                pdf_row.append(p)
            pdf_rows.append(pdf_row)

        col_width = (letter[0] - 2 * inch) / max_cols
        t = Table(pdf_rows, colWidths=[col_width] * max_cols, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), accent_color),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f8")]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        flowables.append(t)
        flowables.append(Spacer(1, 8))
        table_data = []
        in_table = False
        return flowables

    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith('```'):
            if not in_code_block:
                in_code_block = True
                code_lines = []
            else:
                in_code_block = False
                # Flush code block
                if table_data:
                    story.extend(flush_table())
                code_text = '\n'.join(escape_xml(l) for l in code_lines)
                # Split long code into paragraphs
                for cl in code_lines:
                    cl_esc = escape_xml(cl) if cl.strip() else '&nbsp;'
                    story.append(Paragraph(
                        f'<font name="{MONO_FONT}" size="8">{cl_esc}</font>',
                        ParagraphStyle('Code', fontName=MONO_FONT, fontSize=8,
                                       leading=11, leftIndent=18,
                                       backColor=colors.HexColor("#f5f5f5"),
                                       spaceAfter=0)))
                story.append(Spacer(1, 6))
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^---+\s*$', line.strip()):
            if table_data:
                story.extend(flush_table())
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.HexColor("#cccccc"),
                                     spaceAfter=6, spaceBefore=6))
            i += 1
            continue

        # Table rows
        if '|' in line and line.strip().startswith('|'):
            in_table = True
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            table_data.append(cells)
            i += 1
            continue
        else:
            if in_table and table_data:
                story.extend(flush_table())

        # Headings
        if line.startswith('# '):
            text = escape_xml(line[2:].strip())
            text = inline_markup(text)
            story.append(Paragraph(text, h1_style))
            story.append(HRFlowable(width="100%", thickness=1.5,
                                     color=accent_color, spaceAfter=8))
            i += 1
            continue
        if line.startswith('## '):
            text = escape_xml(line[3:].strip())
            text = inline_markup(text)
            story.append(Paragraph(text, h2_style))
            i += 1
            continue
        if line.startswith('### '):
            text = escape_xml(line[4:].strip())
            text = inline_markup(text)
            story.append(Paragraph(text, h3_style))
            i += 1
            continue
        if line.startswith('#### '):
            text = escape_xml(line[5:].strip())
            text = inline_markup(text)
            story.append(Paragraph(f'<b>{text}</b>', body_style))
            i += 1
            continue

        # Bullet points
        if re.match(r'^[-*+] ', line):
            text = escape_xml(line[2:].strip())
            text = inline_markup(text)
            story.append(Paragraph(f'\u2022&nbsp;&nbsp;{text}', bullet_style))
            i += 1
            continue

        # Numbered list
        m = re.match(r'^(\d+)\. (.+)', line)
        if m:
            num = m.group(1)
            text = escape_xml(m.group(2).strip())
            text = inline_markup(text)
            story.append(Paragraph(f'<b>{num}.</b>&nbsp;&nbsp;{text}', bullet_style))
            i += 1
            continue

        # Empty line
        if not line.strip():
            if table_data:
                story.extend(flush_table())
            story.append(Spacer(1, 6))
            i += 1
            continue

        # Regular paragraph
        text = escape_xml(line.strip())
        text = inline_markup(text)
        if text:
            story.append(Paragraph(text, body_style))
        i += 1

    if table_data:
        story.extend(flush_table())

    return story


def build_cover_page(title, accent_color):
    styles = getSampleStyleSheet()
    cover_title = ParagraphStyle('CoverTitle', fontName=BOLD_FONT, fontSize=22,
                                  textColor=accent_color, alignment=TA_CENTER,
                                  spaceAfter=16, leading=28)
    cover_sub = ParagraphStyle('CoverSub', fontName=BASE_FONT, fontSize=12,
                                textColor=colors.HexColor("#444444"),
                                alignment=TA_CENTER, spaceAfter=10, leading=16)
    story = [
        Spacer(1, 1.5 * inch),
        Paragraph("IEEE TNNLS Peer Review Package", cover_sub),
        Spacer(1, 0.3 * inch),
        Paragraph(title, cover_title),
        Spacer(1, 0.2 * inch),
        HRFlowable(width="80%", thickness=2, color=accent_color),
        Spacer(1, 0.3 * inch),
        Paragraph("Adaptive Probabilistic Routing Reinforcement (APRR)", cover_sub),
        Paragraph("Online Policy Iteration for Dynamic Agent-to-Agent Routing", cover_sub),
        Paragraph("in Tool-Augmented LLM Workflows", cover_sub),
        Spacer(1, 0.5 * inch),
        Paragraph("Contains: Review #1 (Harvard CSE) · Review #2 (MIT CSAIL) · Author Response",
                  ParagraphStyle('CoverNote', fontName=ITALIC_FONT, fontSize=10,
                                  textColor=colors.HexColor("#666666"),
                                  alignment=TA_CENTER)),
        PageBreak()
    ]
    return story


def main():
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title="APRR Peer Review Package — IEEE TNNLS",
        author="Peer Review Simulation",
    )

    story = []

    # Cover page
    story.extend(build_cover_page("Peer Review Package", colors.HexColor("#1a237e")))

    file_labels = {
        "review_harvard.md": "Review #1 — Harvard CSE Perspective",
        "review_mit.md": "Review #2 — MIT CSAIL Perspective",
        "author_response.md": "Author Response to Reviewers",
    }

    for fname in FILES:
        fpath = os.path.join(BASE_DIR, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            md_text = f.read()

        accent = TITLE_COLORS[fname]
        # Section divider
        divider_style = ParagraphStyle('Divider', fontName=BOLD_FONT, fontSize=14,
                                        textColor=colors.white, backColor=accent,
                                        spaceAfter=12, spaceBefore=8, leading=20,
                                        leftIndent=0)
        story.append(Paragraph(f'&nbsp;&nbsp;{escape_xml(file_labels[fname])}',
                                divider_style))
        story.append(Spacer(1, 6))

        section_story = parse_markdown_to_story(md_text, accent, styles)
        story.extend(section_story)
        story.append(PageBreak())

    doc.build(story)
    print(f"PDF written to: {OUTPUT}")
    import os as _os
    size = _os.path.getsize(OUTPUT)
    print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")


if __name__ == '__main__':
    main()
