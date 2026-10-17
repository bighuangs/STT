r"""
PDF 解析服务 - 调用 MinerU 云端 API 完成 PDF 到 Markdown 的转换
幂等性分类（多次执行结果相同）

HTTP 请求报文结构（4个部分）:
    1. 请求行 - 方法、URL、协议版本，如 GET /index.html HTTP/1.1
    2. 请求头   - 键值对，如 Host、Content-Type、Authorization、User-Agent
    3. 请求空行     - 用于分隔请求头与请求体，固定为一个空行（\r\n）
    4. 请求体   - 可选，POST/PUT/PATCH 时存放实际数据（json/data/files）

HTTP 方法速查:
    GET   - 获取资源（只读安全，幂等，无请求体）
    POST  - 创建资源（非幂等，支持 json/data/files 参数）
    PUT   - 完整替换资源（幂等，支持 json/data/files）
    PATCH - 部分修改资源（非幂等）
    DELETE- 删除资源（幂等）
    HEAD  - 只获取响应头（检查资源是否存在）
    OPTIONS- 查询支持的 HTTP 方法（CORS 预检）

requests 参数速查:
    url        - str, 请求地址
    params     - dict, URL 查询参数，拼接为 ?key=val
    headers    - dict, 请求头，如 {"Authorization": "Bearer token"}
    json       - dict, 自动序列化 JSON，Content-Type: application/json
    data       - dict/str/bytes, 表单编码，Content-Type: application/x-www-form-urlencoded
    files      - dict, 文件上传，如 {"file": open("x.pdf","rb")}
    timeout    - int 或 (connect, read) 元组，超时秒数
    proxies    - dict, 代理，如 {"http": "http://127.0.0.1:7890"}
    verify     - bool 或证书路径，是否验证 SSL
    auth       - (username, password) 元组，Basic Auth
    cookies    - dict 或 CookieJar
    allow_redirects - bool，是否跟随重定向

响应对象:
    response.status_code - HTTP 状态码
    response.json()      - JSON 响应体
    response.text        - 文本响应
    response.content     - 字节内容
    response.headers     - 响应头
    response.cookies     - 响应 Cookie

正则 re 速查:
    re.compile(pattern)           - 编译正则表达式，返回 Pattern 对象（常用 flags: re.I 忽略大小写, re.S 点号匹配换行, re.M 多行模式）
    re.match(pattern, text)       - 从字符串开头匹配，匹配成功返回 Match 对象，否则 None
    re.search(pattern, text)       - 扫描整个字符串，查找第一个匹配
    re.findall(pattern, text)     - 查找所有匹配，返回字符串列表（无分组时）或元组列表（有分组时）
    re.finditer(pattern, text)     - 返回所有匹配的迭代器，每次迭代得到 Match 对象
    re.sub(pattern, repl, text)    - 替换匹配项，repl 可为字符串或函数
    re.split(pattern, text)        - 按正则分割字符串
    re.fullmatch(pattern, text)    - 整个字符串必须完全匹配

Match 对象方法:
    match.group(0)  - 完整匹配内容
    match.group(1)  - 第 1 个捕获组
    match.group(n)  - 第 n 个捕获组
    match.groups()  - 所有捕获组元组
    match.start()   - 匹配起始位置
    match.end()     - 匹配结束位置
    match.span()    - (start, end) 元组

常用正则符号:
    .      - 任意字符（除换行，re.S 时含换行）
    \\d    - 数字，等价于 [0-9]
    \\D    - 非数字
    \\w    - 字母数字下划线，等价于 [a-zA-Z0-9_]
    \\W    - 非 \\w
    \\s    - 空白字符（空格、\\t、\\n）
    \\S    - 非空白字符
    ^      - 字符串开头
    $      - 字符串结尾
    *      - 0次或多次
    +      - 1次或多次
    ?      - 0次或1次（可选）
    {n}    - 恰好 n 次
    {n,}   - 至少 n 次
    {n,m}  - n 到 m 次
    [abc]  - 字符集，匹配 a 或 b 或 c
    [^abc] - 取反，匹配非 a/b/c
    (abc)  - 捕获组，group() 可提取
    (?:abc)- 非捕获组，只分组不捕获
    a|b    - 匹配 a 或 b
    \b     - 单词边界
    \\    - 转义特殊字符

re技巧:
    编译后使用（性能更好）:
        pattern = re.compile(r'\d+')
        pattern.findall(text)

pathlib.Path 速查:
    构造:
        Path("a/b/c.txt")           - 相对路径
        Path("/home/user/file.txt") - 绝对路径
        Path()                      - 当前目录
        Path(".") / "sub" / "file"  - 用 / 拼接路径
        Path("~").expanduser()      - 展开 ~ 为用户主目录

    路径属性:
        p.name      - 文件名（含后缀），如 "report.pdf"
        p.stem      - 文件名（不含后缀），如 "report"
        p.suffix    - 扩展名，如 ".pdf"
        p.parent    - 父目录路径对象
        p.parents   - 所有祖先目录（迭代器）
        p.parts     - 路径各部分元组，如 ("/", "home", "user")
        p.anchor    - 路径前缀，如 "/" 或 "C:\\"
        p.root      - 根目录，如 "/" 或 "C:\\"
        p.drive     - 盘符，如 "C:"

    路径判断（返回 bool）:
        p.exists()          - 路径是否存在
        p.is_file()          - 是否为文件
        p.is_dir()           - 是否为目录
        p.is_absolute()      - 是否为绝对路径
        p.is_relative_to(x)  - 是否相对路径于 x
        p.match("*.pdf")     - 是否匹配通配符模式

    路径操作:
        p.mkdir(parents=True, exist_ok=True)   - 创建目录（parents=True 可建多级）
        p.unlink()  /  p.rmdir()              - 删除文件 / 空目录
        p.rename("new.txt")                   - 重命名或移动
        p.replace("new.txt")                  - 强制替换
        p.symlink_to("target")                - 创建符号链接
        p.read_text(encoding="utf-8")         - 读取文本内容
        p.read_bytes()                        - 读取字节内容
        p.write_text("content", encoding="utf-8") - 写入文本
        p.write_bytes(b"...")                 - 写入字节
        p.resolve()                           - 返回绝对路径（解析符号链接）
        p.expanduser()                        - 展开 ~
        p.with_suffix(".md")                  - 替换扩展名
        p.with_name("new.pdf")                - 替换文件名
        p.with_stem("report")                 - 替换不含后缀的文件名

    路径遍历:
        p.iterdir()    - 遍历目录下所有文件/子目录
        p.rglob("*.py") - 递归遍历所有 py 文件
        p.glob("*.py")  - 遍历当前目录下所有 py 文件
        list(p.rglob("*.md"))  # -> [Path对象列表]

    Path 对象与 str 互转:
        str(p)         - 转为字符串路径
        Path(str)      - 字符串转为 Path 对象

shutil 速查:
    shutil.copy(src, dst)                          - 复制文件
    shutil.copytree(src, dst)                      - 递归复制目录
    shutil.move(src, dst)                          - 移动/重命名文件或目录
    shutil.rmtree(path)                            - 递归删除目录树
    shutil.make_archive(base_name, format, root_dir) - 创建压缩包（format: zip/tar/gztar）
    shutil.unpack_archive(filename, extract_dir)    - 解压压缩包（自动识别格式）
"""
import hashlib
import os
import shutil
import time
import zipfile
import requests

from app.infra.document_parse.mineru_gateway import mineru_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import step_log,logger,PROJECT_ROOT
from app.rag.import_.config import MINERU_MODEL_VERSION
from pathlib import Path, PurePosixPath
from uuid import uuid4


_MINERU_IDENTITY_DIGEST_LENGTH = 7
_MINERU_ATTEMPT_NONCE_LENGTH = 12
_WINDOWS_SAFE_PATH_BUDGET = 240
_WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CON",
    "NUL",
    "PRN",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


def _build_mineru_artifact_stem(
    state: ImportGraphState,
    pdf_path_obj: Path,
) -> str:
    """为 MinerU 本地制品生成调用唯一、短且与用户文件名解耦的目录名。

    ``local_dir`` 通常已经位于内容寻址的 ``<64-hex>.work`` 目录下；继续使用
    原始 PDF stem（尤其是解密副本的 hash + UUID）会轻易超过 Windows 的传统
    路径预算。任务 ID 在正常导入与预览链路中都由源内容哈希派生；旧调用方若
    没有任务 ID，则退化为对文件 basename 做哈希。7 位身份摘要便于关联，
    12 位随机 nonce 防止同一 PDF 的并发预览争用解压目录。最终名称始终为
    24 个 ASCII 字符，不泄露原文件名，也不会把路径分隔符带入落盘路径。
    """

    task_id = str(state.get("task_id") or "").strip()
    identity = task_id or pdf_path_obj.name
    digest = hashlib.sha256(identity.encode("utf-8", errors="surrogatepass")).hexdigest()
    return (
        f"pdf-{digest[:_MINERU_IDENTITY_DIGEST_LENGTH]}-"
        f"{uuid4().hex[:_MINERU_ATTEMPT_NONCE_LENGTH]}"
    )


def _validate_mineru_artifact_stem(stem: str) -> None:
    """只允许由 ``_build_mineru_artifact_stem`` 产生的受控目录名。"""

    parts = stem.split("-")
    digest = parts[1] if len(parts) == 3 and parts[0] == "pdf" else ""
    nonce = parts[2] if len(parts) == 3 and parts[0] == "pdf" else ""
    if (
        len(digest) != _MINERU_IDENTITY_DIGEST_LENGTH
        or len(nonce) != _MINERU_ATTEMPT_NONCE_LENGTH
        or any(character not in "0123456789abcdef" for character in digest + nonce)
    ):
        raise ValueError("MinerU 本地制品目录名不符合安全格式")


def _safe_extract_mineru_zip(zip_path: Path, target_dir: Path) -> None:
    """在 MinerU 返回值进入本地文件系统前阻断 Zip Slip、炸弹和加密条目。"""

    max_entries = 10_000
    max_total = 1024 * 1024 * 1024
    max_entry = 256 * 1024 * 1024
    max_ratio = 200.0
    root = target_dir.resolve()
    seen: set[str] = set()
    total = 0
    try:
        archive = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("MinerU 返回的 ZIP 已损坏") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > max_entries:
            raise ValueError("MinerU ZIP 条目数超过安全上限")
        for info in infos:
            name = info.filename.replace("\\", "/")
            comparison_name = name.casefold()
            if not name or comparison_name in seen:
                raise ValueError("MinerU ZIP 含空名称或重复条目")
            seen.add(comparison_name)
            parts = PurePosixPath(name).parts
            if name.startswith("/") or any(part in {"", ".", ".."} for part in parts):
                raise ValueError("MinerU ZIP 含不安全路径")
            if len(parts) > 32:
                raise ValueError("MinerU ZIP 路径层级超过安全上限")
            for part in parts:
                reserved_base = part.split(".", 1)[0].upper()
                if (
                    len(part) > 255
                    or ":" in part
                    or part.endswith((" ", "."))
                    or reserved_base in _WINDOWS_RESERVED_NAMES
                ):
                    raise ValueError("MinerU ZIP 含 Windows 不安全文件名")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError("MinerU ZIP 禁止符号链接")
            if info.flag_bits & 0x1:
                raise ValueError("MinerU ZIP 禁止加密条目")
            if info.file_size > max_entry:
                raise ValueError("MinerU ZIP 单条目超过安全上限")
            total += info.file_size
            if total > max_total:
                raise ValueError("MinerU ZIP 总解压大小超过安全上限")
            if info.file_size and info.file_size / max(1, info.compress_size) > max_ratio:
                raise ValueError("MinerU ZIP 压缩比超过安全上限")
            destination = (root / Path(*parts)).resolve()
            if root != destination and root not in destination.parents:
                raise ValueError("MinerU ZIP 解压路径越界")
            if os.name == "nt" and len(str(destination)) > _WINDOWS_SAFE_PATH_BUDGET:
                raise ValueError("MinerU ZIP 解压路径超过 Windows 安全预算")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)

@step_log("validate_pdf_paths")
def validate_pdf_paths(state: ImportGraphState)  -> tuple[Path, Path]:
    """
    校验PDF文件路径与输出目录，确保文件存在、目录可用并自动补全目录
    :param state: 流程状态字典，包含 pdf_path、local_dir 字段
    :return: 元组(解析后的PDF路径对象, 本地输出目录对象)
    :raises ValueError: pdf_path 为空时抛出异常
    :raises FileNotFoundError: PDF文件不存在时抛出异常
    """
    # 1.从状态中读取PDF文件路径和本地输出目录
    pdf_path = state.get("pdf_path")
    local_dir = state.get("local_dir")

    # 2. 校验PDF文件路径和本地输出目录是否可用
    if not pdf_path:
        logger.error("PDF文件路径为空，请检查输入参数")
        raise ValueError("PDF文件路径为空，请检查输入参数")

    # 3. 输出目录为空时，使用项目根目录下的output目录作为默认目录
    if not local_dir:
        logger.warning("输出目录为空，使用项目根目录下的output目录作为默认目录")
        local_dir = PROJECT_ROOT / "output"
        state["local_dir"] = str(local_dir)

    # 4. 转为Path对象，方便后续路径操作
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    # 5. 校验PDF文件是否真实存在，不存在则抛出异常
    if not pdf_path_obj.exists():
        logger.error(f"pdf_path:{pdf_path_obj},但是没有文件存在!")
        raise FileNotFoundError(f"pdf_path:{pdf_path_obj},但是没有文件存在!")

    # 6. 输出目录不存在则自动创建（支持多级目录，已存在也不报错）
    if not local_dir_obj.exists():
        logger.warning(f"local_dir:{local_dir_obj}地址没有文件夹,我们需要主动创建!")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj    

@step_log("upload_pdf_and_poll")
def upload_pdf_and_poll(
    pdf_path_obj: Path,
    mineru_poll_timeout: int = 600,
    mineru_poll_interval: int = 3,
    mineru_download_timeout: int = 30,
) -> str:
    """
    上传PDF文件到MinerU服务，并轮询等待解析完成，最终返回解析结果的下载地址

    超时参数从 state["parser_config"] 读取（与 state.py 默认值对齐）：
        - mineru_poll_timeout: 轮询解析状态总超时（秒），默认 600
        - mineru_poll_interval: 轮询间隔（秒），默认 3
        - mineru_download_timeout: HTTP 下载超时（秒），默认 30

    :param pdf_path_obj: 本地PDF文件路径对象
    :param mineru_poll_timeout: 轮询超时（秒）
    :param mineru_poll_interval: 轮询间隔（秒）
    :param mineru_download_timeout: HTTP 下载超时（秒）
    :return: 解析完成后的ZIP压缩包下载地址
    """
    # 1. 校验MinerU服务配置（base_url、api_key必须存在）
    if not mineru_gateway.base_url or not mineru_gateway.api_key:
        logger.error("MinerU服务配置不正确，base_url或api_key为空")
        # 返回未找到异常
        raise FileNotFoundError("MinerU服务配置不正确，base_url或api_key为空")

    # 2. 构造请求头和请求地址
    url = f"{mineru_gateway.base_url}/file-urls/batch"
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {mineru_gateway.api_key}"
    }

    # 3. 构造请求参数：MinerU 会用文件名后缀识别类型，必须保留受控的
    # ``.pdf``。旧实现使用 ``Path.stem``，导致普通文件和空密码规范化副本都
    # 以无扩展名上传并被 MinerU 以 -60002 unsupported file type 拒绝。
    if pdf_path_obj.suffix.lower() != ".pdf":
        raise ValueError("MinerU PDF 上传文件必须保留 .pdf 扩展名")
    payload = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": MINERU_MODEL_VERSION
    }

    # 4. 请求MinerU获取文件预上传地址
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=mineru_download_timeout)

        if response.status_code != 200:
            logger.error(f"请求MinerU获取文件预上传地址失败，状态码: {response.status_code}, 响应: {response.text}")
            raise RuntimeError(f"请求MinerU获取文件预上传地址失败，状态码: {response.status_code}, 响应: {response.text}")

        # 5. 解析返回结果，检验业务状态码
        result_dict = response.json()
        if result_dict["code"] != 0:
            logger.error(f"申请地址网络状态成功!但是业务失败!错误码:{result_dict['code']},失败信息:{result_dict['msg']}")
            raise RuntimeError(f"申请地址网络状态成功!但是业务失败!错误码:{result_dict['code']},失败信息:{result_dict['msg']}")

        # 6. 提取上传的url和任务批次ID
        file_upload_url = result_dict.get("data",{}).get("file_urls")[0]
        batch_id = result_dict.get("data",{}).get("batch_id")
        logger.info(f"已获取 MinerU 文件上传地址，任务批次ID: {batch_id}")

    except Exception as e:
        # Authorization 和预签名 URL 都属于凭证，不得依赖日志后处理兜底脱敏。
        logger.error(
            f"请求MinerU获取文件预上传地址失败: {e},"
            f"url参数:{url},payload参数:{payload}"
        )
        raise RuntimeError(f"请求MinerU获取文件预上传地址失败: {e}")

    # 7. 上传PDF文件到MinerU的存储地址,使用with，用完就关闭session
    try:
        with requests.Session() as session:
            # session.trust_env = False 是告诉 requests ： 不要信任系统环境变量里的代理、证书、认证等配置 。否则可能MinerU会拒绝请求
            # 它的实际含义是：1. 先创建一个会话对象 session 2. 明确关闭“从系统环境自动继承网络配置”3. 再用这个“干净会话”去执行 PUT 上传
            # 上传的地址是, 预签名上传地址 , OSS / S3 / 对象存储直传地址
            session.trust_env = False
            # 使用put，上传更新
            upload_response = session.put(file_upload_url, data=pdf_path_obj.read_bytes())
            if upload_response.status_code != 200:
                raise RuntimeError(f"上传文件失败,返回状态码为:{upload_response.status_code},请检查minerU配置!")
    except Exception as exc:
        # PUT URL 是短期 capability，requests 异常通常会把完整 URL 写入
        # exception message；这里只记录安全的异常类型和 batch_id。
        logger.error(
            "上传文件失败，异常类型:{}，任务批次ID:{}",
            type(exc).__name__,
            batch_id,
        )
        raise RuntimeError("上传文件失败，请检查 MinerU 上传链路") from None

    # 8. 构造轮询查询地址
    poll_url = f"{mineru_gateway.base_url}/extract-results/batch/{batch_id}"
    timeout = mineru_poll_timeout
    interval_time = mineru_poll_interval
    start_time = time.time()

    # 9. 开始轮询任务解析状态
    while True:
        # 超时判断
        if time.time() - start_time > timeout:
            raise TimeoutError(f"轮询获取{batch_id}任务状态超时,耗时:{time.time() - start_time}秒")

        try:
            # 请求轮询接口
            poll_response = requests.get(poll_url, headers=headers)
        except Exception:
            logger.warning("请求出现异常!可以稍后重试!!")
            time.sleep(interval_time)
            continue

        # 网络异常处理：5xx可重试(minerU服务端问题)，其他直接报错
        if poll_response.status_code != 200:
            if 500 <= poll_response.status_code < 600:
                logger.warning(f"有可修复的网络异常,状态码为:{poll_response.status_code},可以稍后重试!!")
                time.sleep(interval_time + 2)
                continue
            logger.error(f"不可修复的网络状态异常,状态码为:{poll_response.status_code}")
            raise RuntimeError(f"不可修复的网络状态异常,状态码为:{poll_response.status_code}")

        # 解析返回结果
        poll_response_dict = poll_response.json()
        if poll_response_dict["code"] != 0:
            logger.error(f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}")
            raise RuntimeError(
                f"轮询业务异常,错误码:{poll_response_dict['code']},失败信息:{poll_response_dict['msg']}"
            )

        # 10. 获取解析任务状态
        extract_result = poll_response_dict.get("data",{}).get("extract_result")[0]
        extract_result_state = extract_result.get("state")

        # 任务完成返回下载地址,解析失败zip地址为空抛出异常
        if extract_result_state == "done":
            extract_result_url = extract_result.get("full_zip_url")
            if not extract_result_url:
                logger.error("已经完成了解析,但是zip地址为空!!")
                raise RuntimeError("已经完成了解析,但是zip地址为空!!")
            return extract_result_url

        # 任务失败抛出异常
        if extract_result_state == "failed":
            raise RuntimeError(f"已经完成了解析,但是失败了!!失败信息:{extract_result['err_msg']}")

        # 任务仍在处理中 → 等待后继续轮询
        logger.warning(f"解析正在进行中,状态:{extract_result_state}!")
        time.sleep(interval_time)

@step_log("download_and_extract_markdown")
def download_and_extract_markdown(
    zip_url: str,
    local_dir_path_obj: Path,
    stem: str,
    mineru_download_timeout: int = 30,
) -> Path:
    """
    下载 MinerU 解析完成的 ZIP 压缩包，解压并提取出标准的 MD 文件
    1. 从 zip_url 下载解析结果压缩包
    2. 解压到指定目录
    3. 自动查找最合适的 MD 文件（优先同名 → full.md → 第一个）
    4. 重命名为统一规范的文件名并返回

    Args:
        zip_url: MinerU 返回的 ZIP 下载地址
        local_dir_path_obj: 本地存放解压文件的目录
        stem: 由 ``_build_mineru_artifact_stem`` 生成的受控短名称
        mineru_download_timeout: HTTP 下载超时（秒），默认 30

    Returns:
        Path: 最终整理好的 MD 文件路径对象
    """
    _validate_mineru_artifact_stem(stem)
    local_dir_path_obj.mkdir(parents=True, exist_ok=True)

    # 1. 从 zip_url 下载解析结果压缩包
    try:
        response = requests.get(zip_url, timeout=mineru_download_timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        status_hint = f"，HTTP {status_code}" if status_code is not None else ""
        raise RuntimeError(f"MinerU ZIP 下载失败{status_hint}") from None
    if len(response.content) > 512 * 1024 * 1024:
        raise ValueError("MinerU 返回的 ZIP 超过下载安全上限")
    # 拼接 ZIP 保存路径：输出目录 + 文件名_result.zip
    zip_path_obj = local_dir_path_obj / f"{stem}_result.zip"
    # 将二进制内容写入 ZIP 文件
    zip_path_obj.write_bytes(response.content)   

    # 2. 解压 ZIP 文件到指定目录
    extract_path_obj = local_dir_path_obj / stem
    # 如果解压目录已存在，先删除（防止旧文件干扰）
    if extract_path_obj.exists():
        shutil.rmtree(extract_path_obj)
    # 创建新的解压目录
    extract_path_obj.mkdir(parents=True, exist_ok=True)
    # 安全逐条目解压，禁止路径穿越、符号链接、加密条目和高压缩比炸弹。
    _safe_extract_mineru_zip(zip_path_obj, extract_path_obj)

    # 3. 查找解压后的所有MD文件,若未空则解压失败,此处使用rglob查询递归目录所有.md文件
    md_file_list = list(extract_path_obj.rglob("*.md"))
    if not md_file_list:
        logger.error(f"解压后的MD文件列表为空,解压目录: {extract_path_obj}")
        raise FileNotFoundError(f"解压后的MD文件列表为空,解压目录: {extract_path_obj}")

    # 4. 查找最合适的MD文件,优先同名 → full.md → 第一个

    # 优先级1: 同名MD文件
    for md_file_obj in md_file_list:
        if md_file_obj.stem == stem:
            return md_file_obj
    
    # 优先级2: full.md文件(minerU解析的默认文件名)
    target_md_file_obj = None
    for md_file_obj in md_file_list:
        if md_file_obj.name.lower() == "full.md":
            target_md_file_obj = md_file_obj
            break

    # 优先级3: 若上述都找不到,直接取第一个MD文件
    if not target_md_file_obj:
        target_md_file_obj = md_file_list[0]

    # 5. 重命名为统一规范(和pdf文件名一致)的文件名并返回
    return target_md_file_obj.rename(target_md_file_obj.with_name(f"{stem}.md"))


def _select_mineru_content_list(extract_root: Path) -> Path | None:
    """只接受本次安全解压目录内唯一的 MinerU v1 content-list 制品。

    MinerU ZIP 可能同时包含 ``*_content_list.json`` 和
    ``*_content_list_v2.json``。TODO-08 当前只消费结构稳定的非 v2 文件；候选缺失或
    不唯一时返回 ``None``，由 PDF 适配器明确降级到 Markdown，绝不猜测候选。
    """

    root = extract_root.resolve()
    candidates: list[Path] = []
    for candidate in extract_root.rglob("*.json"):
        lowered = candidate.name.lower()
        if not lowered.endswith("_content_list.json") or lowered.endswith(
            "_content_list_v2.json"
        ):
            continue
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            logger.warning("忽略越出 MinerU 抽取根目录的 content_list 候选: %s", candidate)
            continue
        if resolved.is_file():
            candidates.append(resolved)

    unique = sorted(set(candidates), key=lambda path: path.as_posix().casefold())
    if len(unique) == 1:
        return unique[0]
    if not unique:
        logger.warning("MinerU 抽取结果未提供非 v2 *_content_list.json，将降级到 Markdown")
    else:
        logger.warning(
            "MinerU 抽取结果含 %s 个非 v2 content_list 候选，拒绝猜测并降级到 Markdown: %s",
            len(unique),
            [str(path) for path in unique],
        )
    return None


@step_log("parse_pdf_to_markdown")
def parse_pdf_to_markdown(state: ImportGraphState) -> ImportGraphState:
    """
    PDF 解析服务：
    1. 校验 PDF 路径和输出目录是否可用
    2. 调用 MinerU 上传 PDF 并用batch_id轮询解析结果,采用“云端 API”模式，将繁重的解析任务卸载到 MinerU 服务端。
    3. 下载并解压解析结果解析完成后，不仅提取 Markdown 文本，还同步下载并解压关联的图片资源，保持文档的多模态完整性
    4. 获取 Markdown 路径和正文内容
    5. 回写 md_path / md_content / local_dir,
    """

    # 1. 校验 PDF 路径和输出目录是否可用
    pdf_path_obj, local_dir_path_obj = validate_pdf_paths(state)

    # 2. 调用 MinerU 上传 PDF 并轮询解析结果,采用“云端 API”模式
    #    超时参数从 parser_config 读取（与 state.py 默认值对齐）
    parser_config = state.get("parser_config", {}) or {}
    mineru_poll_timeout = int(parser_config.get("mineru_poll_timeout", 600))
    mineru_poll_interval = int(parser_config.get("mineru_poll_interval", 3))
    mineru_download_timeout = int(parser_config.get("mineru_download_timeout", 30))

    zip_url = upload_pdf_and_poll(
        pdf_path_obj,
        mineru_poll_timeout=mineru_poll_timeout,
        mineru_poll_interval=mineru_poll_interval,
        mineru_download_timeout=mineru_download_timeout,
    )
    logger.info("MinerU 上传 PDF 并轮询解析结果完成，已取得 ZIP 下载地址")

    # 3. 下载并解压解析结果解析完成后，不仅提取 Markdown 文本，还同步下载并解压关联的图片资源
    artifact_stem = _build_mineru_artifact_stem(state, pdf_path_obj)
    md_path_obj = download_and_extract_markdown(
        zip_url,
        local_dir_path_obj,
        artifact_stem,
        mineru_download_timeout=mineru_download_timeout,
    )

    state["md_path"] = str(md_path_obj)
    state["md_content"] = md_path_obj.read_text(encoding="utf-8")

    # content_list 与 Markdown 必须来自同一次、同一安全解压根；不沿用 state 中的旧值。
    state.pop("mineru_content_list_path", None)
    extract_root = local_dir_path_obj / artifact_stem
    content_list_path = _select_mineru_content_list(extract_root)
    if content_list_path is not None:
        state["mineru_content_list_path"] = str(content_list_path)

    return state
