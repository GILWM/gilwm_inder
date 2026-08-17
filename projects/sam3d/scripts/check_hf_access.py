"""Check gated Meta model access without printing credentials."""

import os
import urllib.error
import urllib.request


TOKEN_PATH = os.environ.get(
    "HF_TOKEN_FILE",
    "/datassd/morka/cosmos-sam3d-work/.secrets/hf_token",
)
REPOSITORIES = ("facebook/sam3", "facebook/sam-3d-objects")


def main() -> None:
    with open(TOKEN_PATH, encoding="utf-8") as token_file:
        tokens = [line.strip() for line in token_file if line.strip()]
    for repository in REPOSITORIES:
        any_success = False
        for credential_index, token in enumerate(tokens, 1):
            try:
                request = urllib.request.Request(
                    f"https://huggingface.co/{repository}/resolve/main/.gitattributes",
                    headers={"Authorization": f"Bearer {token}"},
                    method="HEAD",
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    status = response.status
                print(f"FILE_ACCESS_OK {repository} credential={credential_index} status={status}")
                any_success = True
            except urllib.error.HTTPError as error:
                print(f"FILE_ACCESS_DENIED {repository} credential={credential_index} status={error.code}")
            except urllib.error.URLError as error:
                print(
                    f"FILE_ACCESS_NETWORK_ERROR {repository} credential={credential_index} "
                    f"reason={type(error.reason).__name__}"
                )
        if not any_success:
            print(f"NO_AUTHORIZED_CREDENTIAL {repository}")


if __name__ == "__main__":
    main()
