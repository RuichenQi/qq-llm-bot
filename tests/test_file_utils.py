"""file_utils: classification + plain-text / code extraction + truncation."""
from __future__ import annotations

from bot import file_utils


def test_classify_extensions():
    assert file_utils.classify("notes.txt") == "text"
    assert file_utils.classify("script.py") == "code"
    assert file_utils.classify("README.md") == "text"
    assert file_utils.classify("paper.pdf") == "pdf"
    assert file_utils.classify("doc.docx") == "docx"
    assert file_utils.classify("clip.mp4") == "video"
    assert file_utils.classify("voice.mp3") == "audio"
    assert file_utils.classify("weird.xyz") == "unsupported"
    # Case-insensitive.
    assert file_utils.classify("PAPER.PDF") == "pdf"


def test_extract_text_plain_utf8():
    extr = file_utils.extract_text(
        "hello.txt", "你好世界\n第二行".encode("utf-8"), max_chars=1000,
    )
    assert extr.kind == "text"
    assert "你好世界" in extr.text
    assert not extr.truncated


def test_extract_text_truncates():
    payload = ("a" * 200).encode("utf-8")
    extr = file_utils.extract_text("big.txt", payload, max_chars=50)
    assert extr.truncated
    assert len(extr.text) == 50


def test_extract_code_wraps_in_fence():
    src = "def hello():\n    print('hi')\n".encode("utf-8")
    extr = file_utils.extract_text("hello.py", src, max_chars=1000)
    assert extr.kind == "code"
    assert extr.text.startswith("```py")
    assert extr.text.endswith("```")
    assert "def hello" in extr.text


def test_extract_unsupported():
    extr = file_utils.extract_text("data.xyz", b"junk", max_chars=100)
    assert extr.kind == "unsupported"
    assert extr.error and "不支持" in extr.error


def test_format_for_prompt_normal():
    extr = file_utils.FileExtraction(
        kind="text", name="notes.txt", text="hello",
    )
    block = file_utils.format_for_prompt(extr)
    assert "[用户上传的文件: notes.txt]" in block
    assert "hello" in block


def test_format_for_prompt_truncated_marker():
    extr = file_utils.FileExtraction(
        kind="text", name="big.txt", text="abc", truncated=True,
    )
    block = file_utils.format_for_prompt(extr)
    assert "内容已截断" in block


def test_format_for_prompt_error():
    extr = file_utils.FileExtraction(
        kind="pdf", name="x.pdf", text="", error="读取失败",
    )
    block = file_utils.format_for_prompt(extr)
    assert "x.pdf" in block
    assert "读取失败" in block


def test_safe_name_strips_path_chars():
    assert file_utils.safe_name("../../etc/passwd") == ".._.._etc_passwd"
    assert file_utils.safe_name("") == "file"
    assert file_utils.safe_name("中文 名字.txt").endswith(".txt")
