"""生成自签 TLS 证书（在游戏机 / relay 侧运行一次）。

生成后：
    certs/key.pem   只留在游戏机（relay 加载）
    certs/cert.pem  复制到控制机（KbdClient 校验服务端用）
"""

import datetime
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
    os.makedirs(out_dir, exist_ok=True)

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

    with open(os.path.join(out_dir, "key.pem"), "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))
    with open(os.path.join(out_dir, "cert.pem"), "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"证书已生成到 {out_dir}/")
    print("key.pem 只留在游戏机；cert.pem 复制到控制机")


if __name__ == "__main__":
    main()
