"""One-time VAPID key pair generator for Web Push.

Run: python backend/scripts/generate_vapid_keys.py
Then set the two printed values as VAPID_PUBLIC_KEY / VAPID_PRIVATE_KEY in .env.
"""

import base64

from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from py_vapid import Vapid02 as Vapid


def _b64url(raw_bytes):
    return base64.urlsafe_b64encode(raw_bytes).rstrip(b"=").decode()


def main():
    vapid = Vapid()
    vapid.generate_keys()

    public_raw = vapid.public_key.public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    private_raw = vapid.private_key.private_numbers().private_value.to_bytes(32, "big")

    print("VAPID_PUBLIC_KEY=", _b64url(public_raw))
    print("VAPID_PRIVATE_KEY=", _b64url(private_raw))


if __name__ == "__main__":
    main()
