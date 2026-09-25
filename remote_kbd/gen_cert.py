"""生成自签 TLS 证书（在游戏机 / relay 侧运行一次）。

生成后：
    certs/key.pem   只留在游戏机（relay 加载）
    certs/cert.pem  复制到控制机（KbdClient 校验服务端用）

**两种用法，同一份实现**：
    python -m remote_kbd.gen_cert            命令行
    部署台的「键盘中继」卡片 → 「生成证书…」   一次点完（A 机原则上不敲命令）

缺 `cryptography` 时**不用急着装** —— 两边要的只是"同一份 cert.pem"，
游戏机上那份已经在用的话，拷到控制机即可（见下面的提示）。
"""

import datetime
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
except ImportError:                           # noqa: BLE001
    # 只有**生成**证书需要 cryptography：relay 与 KbdClient 都只用标准库 ssl。
    # 所以缺它通常**根本不用装** —— 两边要的只是"同一份 cert.pem"，
    # 游戏机上那份已经在用的话，拷到控制机即可（见本文件开头那两行）。
    raise SystemExit(
        "缺 cryptography 这个包（**只有生成证书才需要**）：\n"
        "    python -m pip install cryptography\n\n"
        "但多数情况不用生成 —— 两边的证书**只要一致**就行：\n"
        "  把游戏机上正在用的 remote_kbd/certs/cert.pem 拷到控制机同路径覆盖，\n"
        "  私钥 key.pem 留在游戏机（控制机从不读它，别拷过去）。")


def out_dir():
    """证书默认放哪儿（`remote_kbd/certs`）—— 命令行与部署台按钮**共用同一份**。"""
    return Path(__file__).resolve().parent / "certs"


def generate(target=None):
    """生成一对自签证书 → (cert.pem 路径, key.pem 路径)。

    **唯一一处生成逻辑**：命令行 `main()` 和部署台的「生成证书…」按钮都调它 ——
    两处各写一遍必然漂移（而漂移在这里的后果是"键盘连不上，还看不出为什么"）。

    覆盖已有证书要由**调用方**先确认：那会让控制机手里那份 cert.pem 立刻作废。
    """
    d = Path(target) if target else out_dir()
    d.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "kbd-relay"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .sign(key, hashes.SHA256())
    )

    key_p = d / "key.pem"
    cert_p = d / "cert.pem"
    key_p.write_bytes(key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption(),
    ))
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return cert_p, key_p


def main():
    cert_p, key_p = generate()
    print("证书已生成到 %s/" % cert_p.parent)
    print("key.pem 只留在游戏机；cert.pem 复制到控制机")

    # 顺手按 relay 的用法加载一次：生成成功 ≠ 这对能被 TLS 用（比如文件写坏了）。
    # 当场验比现场查便宜得多（现场的现象只是"键盘连不上"）。
    import ssl

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(str(cert_p), str(key_p))
    print("已用 TLS 加载验证通过（relay 就是这么加载的）")


if __name__ == "__main__":
    main()
