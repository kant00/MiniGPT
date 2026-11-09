"""
获取中文 Wikipedia 文章，并保存为 data/wiki_zh.jsonl。

默认数据源：
  Hugging Face 的 wikimedia/wikipedia 数据集，配置为 20231101.zh。
  使用 streaming=True 按需读取已经清洗好的 Parquet 数据，无需先下载
  完整数据集，也无需进行 BZ2 解压、XML 解析和 Wikitext 清洗。
  默认获取 100000 篇，过滤正文少于 50 个字符的短文章。

备用数据源：
  1. --source wikimedia
     从 Wikimedia 官方 dump 流式下载原始 BZ2 文件，边下载边解压和解析。
     这种方式处理步骤较多，速度通常慢于默认数据源。
  2. --from_local
     读取 data/raw/*.txt；以空行作为段落分隔，将每段保存为一条数据。

输出格式：
  每行一个 JSON 对象，包含 id、title 和 text 三个字段，例如：
  {"id": "123", "title": "人工智能", "text": "正文内容……"}

文件保护：
  在线获取时先写入 data/wiki_zh.jsonl.part，成功结束后再替换正式文件，
  避免网络中断时损坏已有的 data/wiki_zh.jsonl。

使用方式（推荐在项目根目录执行）：
  python -m src.download_data                         # 默认获取 100000 篇
  python -m src.download_data --max_articles 10000    # 获取 10000 篇
  python -m src.download_data --hf_mirror             # 官网无法连接时使用镜像
  python -m src.download_data --source wikimedia      # 使用 Wikimedia 原始 dump
  python -m src.download_data --from_local            # 使用本地 TXT 文件
  python -m src.download_data --config configs/default.json

也兼容进入 src 目录后直接执行 python download_data.py。
未显式传入的参数从配置文件的 download_data 分区读取。

依赖：
  默认数据源：python -m pip install -U datasets "huggingface_hub>=1.27.0,<2"
  Wikimedia 备用源：python -m pip install requests
"""

import os
import bz2
import re
import json
import argparse
import logging
import time
from pathlib import Path

import xml.etree.ElementTree as ET

try:
    from .config_utils import parse_args_with_config
except ImportError:  # 兼容在 src 目录中直接运行脚本
    from config_utils import parse_args_with_config

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

HF_DATASET = "wikimedia/wikipedia"
HF_CONFIG = "20231101.zh"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

DUMP_URL = (
    "https://dumps.wikimedia.org/zhwiki/latest/"
    "zhwiki-latest-pages-articles1.xml-p1p187712.bz2"
)
MEDIAWIKI_NS = "{http://www.mediawiki.org/xml/export-0.11/}"


# ── Wikimedia 备用源使用的 Wikitext 清理 ─────────────────────────────────────


# 用途：用有序正则清理百科正文中的模板、脚注、表格、链接标记和多余空白。
# 调用时机：Wikimedia 原始 XML 路径提取正文后调用，Hugging Face 已清洗路径不调用。
# 输入输出与注意：输入原始字符串，返回清理文本；链接保留显示文字，模板最多去除五轮，是启发式清洗而非完整语法解析器。
def clean_wikitext(text: str) -> str:
    """
    去除 Wikipedia 标记语言中的格式标记，保留纯中文文本

    处理顺序：
      1. 去除 {{模板}} ：多轮去除，处理嵌套
      2. 去除 [[链接|显示文字]] → 保留显示文字
      3. 去除 [外链 显示] → 保留显示文字
      4. 去除 HTML 标签 <ref>...</ref>、<br/>等
      5. 去除 ==章节标题==
      6. 去除 '''粗体''' ''斜体''
      7. 去除表格（{|...|} 块）
      8. 清理多余空白
    """
    # 去 HTML 注释
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # 去 <ref>...</ref> 脚注
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>]*/>", "", text)
    # 去 {{模板}}（多轮展开，处理嵌套，最多 5 轮）
    for _ in range(5):
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
        if text == prev:
            break
    # 去表格
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)
    # 去 [[文件:xxx]] [[File:xxx]] [[Image:xxx]] 等非正文链接
    text = re.sub(
        r"\[\[(?:文件|File|Image|Category|分类|Special|特殊):[^\]]+\]\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # 展开 [[链接|显示文字]] → 显示文字
    text = re.sub(r"\[\[(?:[^|\]]+\|)?([^\]]+)\]\]", r"\1", text)
    # 去 [外链 显示] → 显示
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    # 去 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去 == 章节标题 ==
    text = re.sub(r"={2,}[^=\n]*={2,}", "\n", text)
    # 去 '''粗体''' ''斜体''
    text = re.sub(r"'{2,3}", "", text)
    # 去行首的表格分隔符（单独一行的 |、!、|-）
    text = re.sub(r"^\s*[|!].*$", "", text, flags=re.MULTILINE)
    # 清理多余空白
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ── Hugging Face 已清洗数据（默认，速度更快）─────────────────────────────────


# 用途：检查文章文件存在、非空、逐行可解析并具备必要字段。
# 调用时机：下载前判断旧数据能否复用，临时文件发布前验证完整性，格式测试也调用。
# 输入输出与注意：返回 (有效标志,已计数行数,原因)；可要求 expected_rows，正文须为非空字符串，不做训练级质量筛选。
def validate_jsonl(path: Path, expected_rows: int = None):
    """校验文件非空、逐行可解析且包含训练需要的字段。"""
    if not path.exists():
        return False, 0, "文件不存在"
    if path.stat().st_size == 0:
        return False, 0, "文件为空"

    rows = 0
    try:
        with open(path, "r", encoding="utf-8") as fin:
            for line_no, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict) or not all(
                    key in obj for key in ("id", "title", "text")
                ):
                    return False, rows, f"第 {line_no} 行缺少 id/title/text 字段"
                if not isinstance(obj["text"], str) or not obj["text"].strip():
                    return False, rows, f"第 {line_no} 行正文为空"
                rows += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, rows, f"读取失败：{exc}"

    if rows == 0:
        return False, 0, "没有有效文章"
    if expected_rows is not None and rows != expected_rows:
        return False, rows, f"期望 {expected_rows} 行，实际 {rows} 行"
    return True, rows, "OK"


# 用途：清理上次未完成的临时文章文件，为重新写入准备。
# 调用时机：各下载/本地导入入口开始处理前调用。
# 输入输出与注意：仅指定 part 存在时删除并记录日志，不删除正式 JSONL；因此本项目下载不是断点续传。
def reset_part_file(part_path: Path) -> None:
    if part_path.exists():
        logger.warning(f"清理未完成的临时文件：{part_path}")
        part_path.unlink()


# 用途：从依赖版本字符串开头解析前三段整数。
# 调用时机：download_hf_articles 执行代码内预设版本检查前调用。
# 输入输出与注意：返回三元组或 None；不完整实现所有版本规范，版本限制的含义由调用方判断。
def parse_version_triplet(version: str):
    """解析版本号的前三段数字，无法解析时返回 None。"""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())


# 用途：从指定中文 Wikipedia 数据集流式获取文章并发布 JSONL。
# 调用时机：download_data.main 的默认在线分支调用。
# 输入输出与注意：输入文章上限和镜像开关，返回输出 Path；复用足量有效旧文件，否则写 part、校验再替换，可能联网且不执行分词。
def download_hf_articles(max_articles: int = 100000, hf_mirror: bool = False) -> Path:
    """从 Hugging Face 流式读取已清洗的中文 Wikipedia 文章。"""
    if hf_mirror:
        # 必须在导入 Hugging Face 库之前设置 HF_ENDPOINT。
        os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT

    try:
        import datasets
        import huggingface_hub
    except ImportError as exc:
        raise ImportError("请先安装 datasets：python -m pip install datasets") from exc

    hub_version = huggingface_hub.__version__
    parsed_hub_version = parse_version_triplet(hub_version)
    if (
        parsed_hub_version is not None
        and (1, 0, 0) <= parsed_hub_version < (1, 27, 0)
    ):
        raise RuntimeError(
            f"检测到 huggingface_hub=={hub_version}，这个版本存在已知的 HTTP 重试错误。\n"
            "请在当前 Python 环境中升级：\n"
            "python -m pip install -U datasets "
            "\"huggingface_hub>=1.27.0,<2\"\n"
            "升级完成后，请重启终端或 Python 内核。"
        )

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    logger.info(
        f"依赖版本：datasets={datasets.__version__}, "
        f"huggingface_hub={hub_version}"
    )
    logger.info(f"Hugging Face 访问地址：{endpoint}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "wiki_zh.jsonl"
    part_path = DATA_DIR / "wiki_zh.jsonl.part"

    existing_valid, existing_rows, existing_reason = validate_jsonl(out_path)
    if existing_valid and existing_rows >= max_articles:
        logger.info(
            f"数据文件已有 {existing_rows} 篇，已达到目标 "
            f"{max_articles} 篇：{out_path}"
        )
        return out_path
    if existing_valid and existing_rows:
        logger.info(
            f"发现未达到目标的旧文件（{existing_rows} 篇），重新获取；"
            "新文件完成前会保留旧文件。"
        )
    elif out_path.exists():
        logger.warning(f"现有数据文件无效（{existing_reason}），将重新获取")

    reset_part_file(part_path)

    logger.info(f"数据源：{HF_DATASET} / {HF_CONFIG}")
    logger.info(f"目标：前 {max_articles} 篇文章 → {out_path}")
    logger.info("流式读取已清洗数据，无需下载完整数据集...")

    start_time = time.monotonic()
    written = 0

    try:
        dataset = datasets.load_dataset(
            HF_DATASET,
            HF_CONFIG,
            split="train",
            streaming=True,
        )

        with open(part_path, "w", encoding="utf-8") as fout:
            for article in dataset:
                text = (article.get("text") or "").strip()
                if len(text) < 50:
                    continue

                fout.write(
                    json.dumps(
                        {
                            "id": str(article.get("id") or written),
                            "title": (article.get("title") or "").strip(),
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

                if written % 1000 == 0:
                    elapsed = max(time.monotonic() - start_time, 0.001)
                    speed = written / elapsed
                    remaining_minutes = (
                        (max_articles - written) / max(speed, 0.001) / 60
                    )
                    logger.info(
                        f"  已写入 {written} 篇，速度 {speed:.1f} 篇/秒，"
                        f"预计剩余 {remaining_minutes:.1f} 分钟"
                    )

                if written >= max_articles:
                    break
    except Exception as exc:
        part_path.unlink(missing_ok=True)
        if not hf_mirror:
            logger.error(
                "huggingface.co 无法连接，请改用： "
                "python download_data.py --hf_mirror"
            )
        raise RuntimeError(
            f"Hugging Face 数据获取失败：{exc}\n\n"
            "可尝试 Wikimedia 备用源：\n"
            "  python download_data.py --source wikimedia\n"
            "或使用本地文本：\n"
            "  python download_data.py --from_local"
        ) from exc

    if written == 0:
        part_path.unlink(missing_ok=True)
        raise RuntimeError("数据源没有返回可用文章，未替换现有数据文件。")

    # 完整写入后再替换目标文件，避免中途失败破坏已有数据。
    valid, rows, reason = validate_jsonl(part_path, expected_rows=written)
    if not valid:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"下载结果校验失败（{reason}），未替换现有数据文件")
    part_path.replace(out_path)

    if written < max_articles:
        logger.warning(f"数据集已读取完，但只获得 {written} 篇文章")
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"完成！共写入 {written} 篇 → {out_path}")
    logger.info(f"文件大小：{size_mb:.1f} MB")
    return out_path


# ── Wikimedia 原始 dump（备用）──────────────────────────────────────────────


# 用途：流式下载 BZ2 压缩百科 XML、逐页清洗并输出文章。
# 调用时机：main 选择 source=wikimedia 备用源时调用。
# 输入输出与注意：分块解压/增量解析，只保留主命名空间非重定向且正文足够长的页面；达到文章上限停止，发布前校验 JSONL。
def download_wiki_articles(max_articles: int = 100000) -> Path:
    """
    流式下载 bz2 压缩的 Wikipedia XML dump，边下载边解析

    流程：
      HTTP stream → 1MB chunks → BZ2Decompressor → XMLPullParser
                                   (分块解压)        (增量解析)
    内存中同时最多存一篇文章的 XML 元素，与文件总大小无关。
    """
    try:
        import requests
    except ImportError:
        raise ImportError("请先安装 requests：pip install requests")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "wiki_zh.jsonl"
    part_path = DATA_DIR / "wiki_zh.jsonl.part"

    existing_valid, existing_rows, existing_reason = validate_jsonl(out_path)
    if existing_valid and existing_rows >= max_articles:
        logger.info(f"数据文件已有 {existing_rows} 篇：{out_path}，跳过下载")
        return out_path
    if out_path.exists() and not existing_valid:
        logger.warning(f"现有数据文件无效（{existing_reason}），将重新获取")
    reset_part_file(part_path)

    logger.info(f"下载源：{DUMP_URL}")
    logger.info(f"目标：前 {max_articles} 篇文章 → {out_path}")
    logger.info("流式下载中（无需等待全部下载完成）...")

    decompressor = bz2.BZ2Decompressor()
    # XMLPullParser：只关注 end 事件，page 结束时再处理整个元素
    parser = ET.XMLPullParser(["end"])

    written = 0
    downloaded_bytes = 0
    chunk_size = 1024 * 1024  # 1MB，减少 Python 循环开销

    try:
        resp = requests.get(
            DUMP_URL,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (educational project)"},
            timeout=60,
        )
        resp.raise_for_status()

        with open(part_path, "w", encoding="utf-8") as fout:
            for raw_chunk in resp.iter_content(chunk_size=chunk_size):
                if not raw_chunk:
                    continue

                # BZ2 解压此块
                try:
                    xml_chunk = decompressor.decompress(raw_chunk)
                except EOFError:
                    break  # BZ2 流正常结束

                if not xml_chunk:
                    continue

                downloaded_bytes += len(raw_chunk)
                parser.feed(xml_chunk)

                # 处理所有已完整解析的元素
                for event, elem in parser.read_events():
                    # 去命名空间前缀，方便判断
                    tag = elem.tag.replace(MEDIAWIKI_NS, "")

                    if tag != "page":
                        continue

                    # 只取主命名空间（ns=0），跳过 Talk/User/Template 等页
                    ns_elem = elem.find(f"{MEDIAWIKI_NS}ns")
                    if ns_elem is None or ns_elem.text != "0":
                        elem.clear()
                        continue

                    title_elem = elem.find(f"{MEDIAWIKI_NS}title")
                    title = (
                        (title_elem.text or "").strip()
                        if title_elem is not None
                        else ""
                    )

                    # 找 text 节点（在 revision 下）
                    text_elem = elem.find(f".//{MEDIAWIKI_NS}text")
                    raw_text = (
                        (text_elem.text or "").strip() if text_elem is not None else ""
                    )

                    # 跳过重定向
                    if raw_text.upper().startswith("#REDIRECT") or raw_text.startswith(
                        "#重定向"
                    ):
                        elem.clear()
                        continue

                    # 清理 wikitext 标记
                    clean = clean_wikitext(raw_text)

                    # 过滤太短的文章（清洗后有效文本不足 50 字符）
                    if len(clean) < 50:
                        elem.clear()
                        continue

                    fout.write(
                        json.dumps(
                            {
                                "id": str(written),
                                "title": title,
                                "text": clean,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    written += 1

                    # 释放内存
                    elem.clear()

                    if written % 5000 == 0:
                        mb = downloaded_bytes / 1024 / 1024
                        logger.info(f"  已写入 {written} 篇，下载 {mb:.0f}MB...")

                    if written >= max_articles:
                        logger.info(f"已达到 {max_articles} 篇上限，停止下载")
                        resp.close()
                        break

                if written >= max_articles:
                    break

    except Exception as e:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Wikimedia 数据获取失败：{e}\n\n"
            "备用方案：将中文 .txt 文件放入 data/raw/ 后运行：\n"
            "  python download_data.py --from_local"
        ) from e

    valid, rows, reason = validate_jsonl(part_path, expected_rows=written)
    if not valid:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"下载结果校验失败（{reason}），未替换现有数据文件")
    part_path.replace(out_path)
    logger.info(f"完成！共写入 {written} 篇 → {out_path}")
    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"文件大小：{size_mb:.1f} MB")
    return out_path


# ── 本地文件兜底 ──────────────────────────────────────────────────────────────


# 用途：将本地 TXT 按空行分段转换为统一文章 JSONL。
# 调用时机：main 启用 from_local 时调用，适合使用自备语料。
# 输入输出与注意：只读目录直接包含的 txt，UTF-8 忽略坏字节，段内行直接拼接、至少 30 字符；返回文件 Path，不受 max_articles 控制。
def from_local_txt(raw_dir: Path) -> Path:
    """
    从本地 .txt 文件构建 wiki_zh.jsonl（网络不通时的兜底）
    把任意中文 .txt 文件放到 data/raw/ 目录后运行。
    """
    txt_files = list(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"未在 {raw_dir} 找到 .txt 文件。\n"
            "请将中文文本文件（任意来源）放入该目录后重试。"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "wiki_zh.jsonl"
    part_path = DATA_DIR / "wiki_zh.jsonl.part"
    reset_part_file(part_path)

    logger.info(f"从本地读取 {len(txt_files)} 个文件：{[f.name for f in txt_files]}")
    written = 0

    with open(part_path, "w", encoding="utf-8") as fout:
        for txt_path in txt_files:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as fin:
                buffer = []
                for line in fin:
                    line = line.strip()
                    if line:
                        buffer.append(line)
                    elif buffer:
                        text = "".join(buffer)
                        if len(text) >= 30:
                            fout.write(
                                json.dumps(
                                    {
                                        "id": str(written),
                                        "title": txt_path.stem,
                                        "text": text,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            written += 1
                        buffer = []
                if buffer:
                    text = "".join(buffer)
                    if len(text) >= 30:
                        fout.write(
                            json.dumps(
                                {
                                    "id": str(written),
                                    "title": txt_path.stem,
                                    "text": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        written += 1

    valid, rows, reason = validate_jsonl(part_path, expected_rows=written)
    if not valid:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"本地数据校验失败（{reason}），未替换现有数据文件")
    part_path.replace(out_path)
    logger.info(f"完成！从本地文件生成 {written} 段 → {out_path}")
    return out_path


# 用途：合并下载配置与 CLI，并在本地、Wikimedia、Hugging Face 路径中选择一个。
# 调用时机：python -m src.download_data 或作为脚本执行时调用。
# 输入输出与注意：检查 max_articles 为正；from_local 优先于 source，会创建本地 raw 目录，实际下载与文件替换由对应函数执行。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_articles", type=int, default=100000)
    parser.add_argument(
        "--source",
        choices=("huggingface", "wikimedia"),
        default="huggingface",
        help="在线数据源，默认使用速度更快的 huggingface",
    )
    parser.add_argument(
        "--hf_mirror",
        action="store_true",
        help="huggingface.co 无法连接时使用 hf-mirror.com 镜像",
    )
    parser.add_argument("--no_hf_mirror", dest="hf_mirror", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--from_local",
        action="store_true",
        help="使用 data/raw/*.txt 本地文件，跳过网络下载",
    )
    parser.add_argument("--online", dest="from_local", action="store_false", help=argparse.SUPPRESS)
    args = parse_args_with_config(parser, "download_data")

    if args.max_articles < 1:
        parser.error("--max_articles 必须大于 0")

    if args.from_local:
        raw_dir = DATA_DIR / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        from_local_txt(raw_dir)
    elif args.source == "wikimedia":
        download_wiki_articles(args.max_articles)
    else:
        download_hf_articles(args.max_articles, hf_mirror=args.hf_mirror)


if __name__ == "__main__":
    main()
