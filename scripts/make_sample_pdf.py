"""生成一份演示用的中文 PDF（多页），用来验证 PDF 解析链路。

PyMuPDF 自带简体中文字体（fontname="china-s"），不依赖系统里装没装中文字体，
所以在任何 Windows / Linux 机器上都能生成同样的文件。

用法：python scripts/make_sample_pdf.py
输出：data/samples/教务处期末安排通知.pdf
"""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz  # PyMuPDF >= 1.24

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "samples" / "教务处期末安排通知.pdf"

TITLE = "关于 2026—2027 学年第一学期期末考试及相关安排的通知"

BODY = [
    ("h", "各学院、各位同学："),
    ("p", "为做好本学期期末考试组织工作，现将考试时间、考场规则、缓考申请及成绩查询等事项通知如下，请遵照执行。"),
    ("h", "一、考试时间"),
    ("p", "1. 期末考试统一安排在 2027 年 1 月 4 日至 1 月 15 日进行，每天三个时段：8:30—10:30、14:00—16:00、19:00—21:00。"),
    ("p", "2. 公共课考试时间由教务处统一编排，于 2026 年 12 月 10 日前在教务系统公布；专业课考试时间由各学院在 12 月 15 日前报教务处备案。"),
    ("p", "3. 考试周期间图书馆主馆延长开放至 23:00，通宵自习室开放时间为 22:30 至次日 6:00。"),
    ("h", "二、考场规则"),
    ("p", "1. 考生须持学生证和身份证进入考场，两证缺一不得参加考试；证件遗失的，须在考前到所在学院开具带照片的证明。"),
    ("p", "2. 开考 15 分钟后不得入场，开考 30 分钟内不得离场；考试期间原则上不得上卫生间，特殊情况须由监考教师陪同。"),
    ("p", "3. 手机、智能手表、蓝牙耳机等电子设备须关机后放在考场指定位置，带入座位的按违纪处理。"),
    ("p", "4. 闭卷考试不得携带任何书籍、笔记、草稿纸；开卷考试仅允许携带教材和手写笔记，不得携带打印资料和电子设备。"),
    ("h", "三、缓考与重修"),
    ("p", "1. 因病因事不能参加考试的，须在考前通过教务系统提交缓考申请，并附二级甲等以上医院证明或学院签字的说明，逾期不再受理。"),
    ("p", "2. 缓考安排在 2027 年 3 月第二周进行，缓考成绩按卷面成绩记载，不设补考。"),
    ("p", "3. 必修课成绩低于 60 分的须重修，重修不设次数限制；选修课成绩低于 60 分的可重修或改选其他课程，已修课程成绩全部计入平均学分绩点。"),
    ("h", "四、成绩查询与复核"),
    ("p", "1. 考试成绩于 2027 年 1 月 25 日 10:00 起在教务系统公布，学生可凭学号和密码查询。"),
    ("p", "2. 对成绩有异议的，须在成绩公布后 5 个工作日内向开课学院提交书面复核申请，复核仅核对卷面分数合计与登分是否有误，不重新评阅。"),
    ("p", "3. 复核结果在受理后 10 个工作日内书面反馈，成绩确有错误的由开课学院报教务处更正。"),
    ("h", "五、联系方式"),
    ("p", "教务处考试中心：行政楼 B 座 203 室，电话 0769-8888 6633，工作日 8:30—12:00、14:00—17:00。"),
]

PAGE_WIDTH, PAGE_HEIGHT = fitz.paper_size("a4")
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 64, 72, 72
BODY_SIZE, TITLE_SIZE, LINE_HEIGHT = 11, 15, 19
CHARS_PER_LINE = 38  # 一行的汉字数（含全角标点），按 A4 宽度估算


def wrap(text: str, width: int = CHARS_PER_LINE) -> list[str]:
    """按字数折行。中文没有空格，直接按字符数切；英文/数字同样按字符切，够用即可。"""
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def build() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = MARGIN_TOP

    def new_page() -> None:
        nonlocal page, y
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        y = MARGIN_TOP

    def write(line: str, size: int, spacing: int) -> None:
        nonlocal y
        if y + spacing > PAGE_HEIGHT - MARGIN_BOTTOM:
            new_page()
        page.insert_text((MARGIN_X, y), line, fontname="china-s", fontsize=size)
        y += spacing

    for line in wrap(TITLE, 24):
        write(line, TITLE_SIZE, LINE_HEIGHT + 6)
    y += 6

    for kind, text in BODY:
        size, spacing = (12, LINE_HEIGHT + 4) if kind == "h" else (BODY_SIZE, LINE_HEIGHT)
        for line in wrap(text):
            write(line, size, spacing)
        if kind == "h":
            y += 2

    doc.set_metadata({"title": TITLE, "author": "教务处", "subject": "期末考试安排"})
    doc.save(OUTPUT, deflate=True)
    doc.close()
    return OUTPUT


if __name__ == "__main__":
    path = build()
    with fitz.open(path) as pdf:
        print(f"[OK] {path.relative_to(ROOT)}  {pdf.page_count} 页  {path.stat().st_size / 1024:.1f} KB")
        print(pdf[0].get_text()[:120].replace("\n", " / "))
