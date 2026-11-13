#!/usr/bin/env python3
r"""edu/가이드/*.md → 강사 가이드 Word 파일 (.docx). 외부 패키지 없이 표준 라이브러리만 쓴다.

  python3 edu/make_docx.py                  # → edu/강사_가이드.docx

지원: 제목 (#~####), 문단, 굵게 (**), 코드 (`), 목록 (- , 1. , 들여쓰기 한 단계), 표 (|),
코드 블록 (```), 인용 (>), 그림 (![설명](경로)),
수식 한 줄 ($$ … $$: \frac{위}{아래}, x_y · x_{yz} 아래 첨자, \times · \div · \approx). 장 (01_~08_ 파일) 마다 새 쪽에서 시작한다.
Word 로 열면 차례 (목차) 를 업데이트할지 묻는다 → 예.
"""

from __future__ import annotations

import re
import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

EDU = Path(__file__).resolve().parent
SRC = EDU / "가이드" # 가이드 md 경로 수정 
OUT = EDU / "강사_가이드.docx" # 파일명 수정
TITLE = "로봇 팔 비전 실습 — 강사 가이드" # 파일 제목 수정
SUBTITLE = "원데이 8차시 (SO-101 7축 팔 · Orbbec 카메라 · Jetson Orin Nano)" # 파일 부제목 수정
FONT_KO = "맑은 고딕"  # Windows Word 기본 한글 글꼴
FONT_EN = "Calibri"
FONT_CODE = "Consolas"
PAGE_W, PAGE_H, MARGIN = 11906, 16838, 1134  # A4, 여백 2 cm (DXA)
TEXT_W = PAGE_W - 2 * MARGIN
EMU_PER_DXA = 635

## line 22 ~ 25: 파일 경로, 파일명, 파일 제목, 파일 부제목 수정 
##- 하는 것: md에 쓴 글자와 표, 코드, 그림을 한 글자도 바꾸지 않고 Word 서식으로 옮깁니다. 제목은 제목 스타일로, 표는 Word 표로 바꾸는 식입니다.
##- 하지 않는 것: 내용 요약, 문장 다듬기, 빠진 내용 채우기. 이건 AI가 하는 일이라 이 파이썬으로는 못 합니다.
## 따라서 쓰고 싶은 내용을 md에 작성하고, SRC에 경로를 넣습니다.

def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        head = f.read(24)
    w, h = struct.unpack(">II", head[16:24])
    return w, h


# ---------------------------------------------------------------- 글자 (run)

def runs(text: str, *, bold: bool = False, size: int | None = None, color: str | None = None) -> str:
    """**굵게** 와 `코드` 를 나눠 run 들로."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        b, code = bold, False
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            part, b = part[2:-2], True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            part, code = part[1:-1], True
        out.append(run(part, bold=b, code=code, size=size, color=color))
    return "".join(out)


def run(text: str, *, bold=False, code=False, size=None, color=None) -> str:
    # 순서는 Word 규칙 (rFonts → b → color → sz → shd) 을 따른다. 틀리면 Word 가 파일을 고치려 든다.
    rpr = []
    if code:
        rpr.append(f'<w:rFonts w:ascii="{FONT_CODE}" w:hAnsi="{FONT_CODE}" w:eastAsia="{FONT_KO}"/>')
    if bold:
        rpr.append("<w:b/>")
    if color:
        rpr.append(f'<w:color w:val="{color}"/>')
    if size:
        rpr.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    if code:
        rpr.append('<w:shd w:val="clear" w:color="auto" w:fill="EEF1F4"/>')
    rp = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    return f'<w:r>{rp}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def para(inner: str, style: str | None = None, *, extra_ppr: str = "") -> str:
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + extra_ppr
    return f"<w:p><w:pPr>{ppr}</w:pPr>{inner}</w:p>"


PAGE_BREAK = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


# ---------------------------------------------------------------- 수식 (Word 수식, OMML)

MATH_SYMBOLS = {r"\times": "×", r"\div": "÷", r"\approx": "≈", r"\cdot": "·", r"\,": " "}


def _mrun(text: str) -> str:
    """수식 안 글자. 영문 한 글자 (변수) 는 기울이고, 단위 (mm) · 숫자 · 기호 · 한글은 바로 세운다."""
    out = []
    for part in re.split(r"([A-Za-z]+)", text.replace("-", "−")):
        if not part:
            continue
        sty = "" if re.fullmatch(r"[A-Za-z]", part) else '<m:rPr><m:sty m:val="p"/></m:rPr>'
        font = f'<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:eastAsia="{FONT_KO}"/></w:rPr>'
        out.append(f'<m:r>{sty}{font}<m:t xml:space="preserve">{escape(part)}</m:t></m:r>')
    return "".join(out)


def _group(src: str, i: int) -> tuple[str, int]:
    """src[i] 가 '{' 면 짝 '}' 까지, 아니면 한 글자."""
    if src[i] != "{":
        return src[i], i + 1
    depth, j = 0, i
    while j < len(src):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0:
            return src[i + 1:j], j + 1
        j += 1
    raise ValueError(f"수식의 {{ }} 짝이 안 맞습니다: {src}")


_CTRL = ('<m:ctrlPr><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/><w:i/></w:rPr></m:ctrlPr>')


def omml(src: str) -> str:
    """간단한 LaTeX 꼴 → OMML 조각."""
    for k, v in MATH_SYMBOLS.items():
        src = re.sub(re.escape(k) + r"(?![A-Za-z])", v, src)
    out, buf, i = [], "", 0
    while i < len(src):
        if src.startswith("\\frac", i):
            out.append(_mrun(buf)); buf = ""
            num, i = _group(src, i + 5)
            den, i = _group(src, i)
            out.append(f"<m:f><m:fPr>{_CTRL}</m:fPr><m:num>{omml(num)}</m:num><m:den>{omml(den)}</m:den></m:f>")
        elif src[i] == "_" and buf:
            base, buf = buf[-1], buf[:-1]
            out.append(_mrun(buf)); buf = ""
            sub, i = _group(src, i + 1)
            out.append(f"<m:sSub><m:sSubPr>{_CTRL}</m:sSubPr><m:e>{_mrun(base)}</m:e><m:sub>{omml(sub)}</m:sub></m:sSub>")
        else:
            buf += src[i]
            i += 1
    out.append(_mrun(buf))
    return "".join(out)


def math_para(src: str) -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="Body"/></w:pPr><m:oMathPara><m:oMathParaPr><m:jc m:val="left"/></m:oMathParaPr><m:oMath>{omml(src.strip())}</m:oMath></m:oMathPara></w:p>'


# ---------------------------------------------------------------- 블록

class Doc:
    def __init__(self) -> None:
        self.body: list[str] = []
        self.media: list[tuple[str, Path]] = []  # (rId, 파일)

    def table(self, rows: list[list[str]]) -> None:
        n = max(len(r) for r in rows)
        rows = [r + [""] * (n - len(r)) for r in rows]
        # 열 너비: 글자 수에 비례 (최소 비율 보장)
        lens = [max(4, min(40, max(len(r[i]) for r in rows))) for i in range(n)]
        tot = sum(lens)
        widths = [int(TEXT_W * x / tot) for x in lens]
        widths[-1] = TEXT_W - sum(widths[:-1])
        # 기록용 표 (본문 칸 절반 넘게 비었거나 ☐) 는 손으로 쓰게 줄을 높인다 (약 9 mm)
        def _blank(c: str) -> bool:
            return c.strip() in ("", "☐", "① · ②")
        body = [c for r in rows[1:] for c in r[1:]]
        is_form = bool(body) and sum(map(_blank, body)) * 2 > len(body)
        form_row = '<w:trHeight w:val="510"/>' if is_form else ""
        if is_form:
            # 글이 있는 열 (항목 · 기준) 은 합쳐서 35% 까지, 적는 열은 나머지를 똑같이 나눈다
            fill = [all(_blank(r[i]) for r in rows[1:]) for i in range(n)]
            lab = [i for i in range(n) if not fill[i]]
            if lab and len(lab) < n:
                lab_w = min(int(TEXT_W * 0.35), sum(widths[i] for i in lab))
                lab_tot = sum(lens[i] for i in lab)
                for i in lab:
                    widths[i] = int(lab_w * lens[i] / lab_tot)
                rest = (TEXT_W - sum(widths[i] for i in lab)) // (n - len(lab))
                for i in range(n):
                    if fill[i]:
                        widths[i] = rest
                widths[-1] = TEXT_W - sum(widths[:-1])
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        trs = []
        for ri, r in enumerate(rows):
            tcs = []
            for ci, cell in enumerate(r):
                shade = '<w:shd w:val="clear" w:color="auto" w:fill="E3EEF3"/>' if ri == 0 else ""
                tcs.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shade}</w:tcPr>'
                    + para(runs(cell, bold=(ri == 0), size=19), "TableText") + "</w:tc>")
            head = ("<w:trPr><w:tblHeader/><w:cantSplit/></w:trPr>" if ri == 0
                    else f"<w:trPr><w:cantSplit/>{form_row}</w:trPr>")
            trs.append(f"<w:tr>{head}{''.join(tcs)}</w:tr>")
        self.body.append(
            '<w:tbl><w:tblPr><w:tblStyle w:val="Grid"/>'
            f'<w:tblW w:w="{TEXT_W}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
            '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
            f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>")
        self.body.append(para("", "Spacer"))

    def image(self, path: Path, caption: str) -> None:
        rid = f"rIdImg{len(self.media) + 1}"
        self.media.append((rid, path))
        w, h = png_size(path)
        max_w = TEXT_W * EMU_PER_DXA
        cx = max_w
        cy = int(max_w * h / w)
        max_h = int((PAGE_H - 2 * MARGIN) * EMU_PER_DXA * 0.7)
        if cy > max_h:
            cy, cx = max_h, int(max_h * w / h)
        n = len(self.media)
        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{n}" name="그림 {n}"/>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{escape(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            "</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>")
        self.body.append(para(drawing, "Figure"))
        if caption:
            self.body.append(para(runs(caption), "Caption"))

    def add_markdown(self, text: str, base: Path, first: bool) -> None:
        lines = text.splitlines()
        i = 0
        if not first:
            self.body.append(PAGE_BREAK)
        while i < len(lines):
            line = lines[i]
            s = line.strip()
            if s == "<!-- 쪽 나눔 -->":
                # md 에는 안 보이고 Word 에서만 쪽을 넘긴다
                self.body.append(PAGE_BREAK)
                i += 1
                continue
            if not s or s == "---":
                i += 1
                continue
            if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
                self.body.append(math_para(s[2:-2]))
                i += 1
                continue
            if s.startswith("```"):
                i += 1
                code = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code.append(lines[i])
                    i += 1
                i += 1
                for c in code:
                    self.body.append(para(run(c, code=False), "Code"))
                self.body.append(para("", "Spacer"))
                continue
            m = re.match(r"^(#{1,4})\s+(.*)$", s)
            if m:
                lvl = len(m.group(1))
                self.body.append(para(runs(m.group(2)), f"Heading{lvl}"))
                i += 1
                continue
            if s.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                        rows.append(cells)
                    i += 1
                self.table(rows)
                continue
            m = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
            if m:
                self.image((base / m.group(2)).resolve(), m.group(1))
                i += 1
                continue
            if s.startswith(">"):
                quote = []
                while i < len(lines) and lines[i].strip().startswith(">"):
                    quote.append(lines[i].strip().lstrip(">").strip())
                    i += 1
                for q in quote:
                    if q:
                        self.body.append(para(runs(q), "Quote"))
                continue
            m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
            if m:
                indent = 1 if len(m.group(1)) >= 2 else 0
                marker = m.group(2)
                txt = m.group(3)
                if marker in ("-", "*"):
                    ppr = f'<w:numPr><w:ilvl w:val="{indent}"/><w:numId w:val="1"/></w:numPr>'
                    self.body.append(para(runs(txt), "ListPara", extra_ppr=ppr))
                else:
                    left = 420 + 360 * indent
                    ppr = f'<w:ind w:left="{left}" w:hanging="300"/>'
                    self.body.append(para(run(marker + " ") + runs(txt), "ListPara", extra_ppr=ppr))
                i += 1
                continue
            # 보통 문단: 다음 빈 줄까지 이어 붙인다
            buf = [s]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^\s*([-*]|\d+\.|#|\||>|```|!\[)", lines[i]):
                buf.append(lines[i].strip())
                i += 1
            self.body.append(para(runs(" ".join(buf)), "Body"))

    def title_page(self) -> None:
        self.body.append(para("", "Spacer", extra_ppr='<w:spacing w:before="3600"/>'))
        self.body.append(para(runs(TITLE), "Title"))
        self.body.append(para(runs(SUBTITLE), "Subtitle"))
        self.body.append(PAGE_BREAK)
        self.body.append(para(runs("차례"), "TOCHeading"))
        self.body.append(
            '<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
            '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            + run("(Word 로 열어 차례를 업데이트하면 여기에 목차가 나옵니다)") +
            '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>')
        self.body.append(PAGE_BREAK)


# ---------------------------------------------------------------- 패키지 파일

def styles_xml() -> str:
    def font(size, bold=False, color=None, font_en=FONT_EN):
        return (f'<w:rPr><w:rFonts w:ascii="{font_en}" w:hAnsi="{font_en}" w:eastAsia="{FONT_KO}" w:cs="{font_en}"/>'
                + ("<w:b/>" if bold else "") + (f'<w:color w:val="{color}"/>' if color else "")
                + f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>')

    def pstyle(sid, name, ppr, rpr, outline=None, based="Normal"):
        ol = f'<w:outlineLvl w:val="{outline}"/>' if outline is not None else ""
        return (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/>'
                f'<w:basedOn w:val="{based}"/><w:qFormat/><w:pPr>{ppr}{ol}</w:pPr>{rpr}</w:style>')

    st = [
        '<w:docDefaults><w:rPrDefault>' + font(21)[7:-8].join(["<w:rPr>", "</w:rPr>"]) + "</w:rPrDefault>"
        '<w:pPrDefault><w:pPr><w:spacing w:after="80" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>',
        f'<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>{font(21)}</w:style>',
        pstyle("Title", "Title", '<w:spacing w:after="240"/><w:jc w:val="center"/>', font(44, True, "0E4A5C")),
        pstyle("Subtitle", "Subtitle", '<w:spacing w:after="120"/><w:jc w:val="center"/>', font(24, False, "5B6474")),
        pstyle("TOCHeading", "TOC Heading", '<w:spacing w:after="200"/>', font(32, True, "0E4A5C")),
        pstyle("Heading1", "heading 1", '<w:keepNext/>'
               '<w:pBdr><w:bottom w:val="single" w:sz="8" w:space="4" w:color="0E7490"/></w:pBdr>'
               '<w:spacing w:before="120" w:after="200"/>',
               font(34, True, "0E4A5C"), outline=0),
        pstyle("Heading2", "heading 2", '<w:keepNext/><w:spacing w:before="300" w:after="120"/>',
               font(27, True, "0E7490"), outline=1),
        pstyle("Heading3", "heading 3", '<w:keepNext/><w:spacing w:before="220" w:after="80"/>',
               font(23, True, "1F2937"), outline=2),
        pstyle("Heading4", "heading 4", '<w:keepNext/><w:spacing w:before="160" w:after="60"/>',
               font(21, True, "374151"), outline=3),
        pstyle("Body", "Body", "", font(21)),
        pstyle("ListPara", "List Paragraph", '<w:spacing w:after="40"/>', font(21)),
        pstyle("Quote", "Quote", '<w:pBdr><w:left w:val="single" w:sz="18" w:space="8" w:color="0E7490"/></w:pBdr>'
               '<w:shd w:val="clear" w:color="auto" w:fill="F1F7F9"/>'
               '<w:spacing w:after="60"/><w:ind w:left="284" w:right="284"/>', font(21, False, "1F2937")),
        pstyle("Code", "Code", '<w:shd w:val="clear" w:color="auto" w:fill="F3F4F6"/>'
               '<w:spacing w:after="0" w:line="260" w:lineRule="auto"/><w:ind w:left="200"/>', font(18, False, "111827", FONT_CODE)),
        pstyle("TableText", "Table Text", '<w:spacing w:after="0" w:line="260" w:lineRule="auto"/>', font(19)),
        pstyle("Figure", "Figure", '<w:keepNext/><w:spacing w:before="120" w:after="40"/><w:jc w:val="center"/>', font(21)),
        pstyle("Caption", "caption", '<w:spacing w:after="200"/><w:jc w:val="center"/>', font(18, False, "5B6474")),
        pstyle("Spacer", "Spacer", '<w:spacing w:after="60"/>', font(10)),
        '<w:style w:type="table" w:styleId="Grid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders>'
        + "".join(f'<w:{b} w:val="single" w:sz="4" w:space="0" w:color="C9D2DA"/>'
                  for b in ("top", "left", "bottom", "right", "insideH", "insideV"))
        + "</w:tblBorders></w:tblPr></w:style>",
    ]
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            + "".join(st) + "</w:styles>")


NUMBERING = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="hybridMultilevel"/>'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/>'
    '<w:pPr><w:ind w:left="420" w:hanging="280"/></w:pPr></w:lvl>'
    '<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="–"/><w:lvlJc w:val="left"/>'
    '<w:pPr><w:ind w:left="780" w:hanging="280"/></w:pPr></w:lvl>'
    '</w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num></w:numbering>')


def footer_xml() -> str:
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
            '<w:r><w:rPr><w:sz w:val="16"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:rPr><w:sz w:val="16"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
            '<w:r><w:rPr><w:sz w:val="16"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


def build() -> Path:
    doc = Doc()
    doc.title_page()
    files = sorted(p for p in SRC.glob("*.md") if not p.name.startswith("00_"))
    for k, f in enumerate(files):
        doc.add_markdown(f.read_text(encoding="utf-8"), f.parent, first=(k == 0))
    rels = ['<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
            '<Relationship Id="rIdNum" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>',
            '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
            '<Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>']
    for rid, p in doc.media:
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{rid}.png"/>')
    sect = (f'<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/>'
            f'<w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
            f'<w:pgMar w:top="{MARGIN}" w:right="{MARGIN}" w:bottom="{MARGIN}" w:left="{MARGIN}" w:header="567" w:footer="567" w:gutter="0"/></w:sectPr>')
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
                'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
                'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
                f"<w:body>{''.join(doc.body)}{sect}</w:body></w:document>")
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Default Extension="png" ContentType="image/png"/>'
                     '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                     '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
                     '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
                     '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                     '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                     "</Types>")
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                 '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                 "</Relationships>")
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>' + escape(TITLE) + "</dc:title></cp:coreProperties>")
    settings = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:updateFields w:val="true"/></w:settings>')
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/numbering.xml", NUMBERING)
        z.writestr("word/settings.xml", settings)
        z.writestr("word/footer1.xml", footer_xml())
        z.writestr("word/_rels/document.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   + "".join(rels) + "</Relationships>")
        for rid, p in doc.media:
            z.write(p, f"word/media/{rid}.png")
    return OUT


if __name__ == "__main__":
    out = build()
    # XML 이 올바른지 확인
    import xml.dom.minidom

    with zipfile.ZipFile(out) as z:
        for name in z.namelist():
            if name.endswith(".xml") or name.endswith(".rels"):
                xml.dom.minidom.parseString(z.read(name))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")
    sys.exit(0)
