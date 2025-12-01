"""Patch Google ADK to use new A2A Protocol v0.3.16 agent-card.json path.

This script patches the installed google-adk package to use the correct
agent card endpoint path (/.well-known/agent-card.json) instead of the
deprecated path (/.well-known/agent.json).

Usage:
    python deploy/patch_adk_agent_card_path.py

This should be run after installing the google-adk package and before
starting the A2A server.
"""

import sys
import glob
from pathlib import Path


def find_adk_fast_api():
    """Find the fast_api.py file in the installed google-adk package."""
    # First try sys.path
    for path in sys.path:
        fast_api_path = Path(path) / "google" / "adk" / "cli" / "fast_api.py"
        if fast_api_path.exists():
            return fast_api_path

    # Try common uv/pip installation paths
    search_patterns = [
        "/app/.venv/lib/python*/site-packages/google/adk/cli/fast_api.py",
        "~/.local/lib/python*/site-packages/google/adk/cli/fast_api.py",
        "/usr/local/lib/python*/site-packages/google/adk/cli/fast_api.py",
        ".venv/lib/python*/site-packages/google/adk/cli/fast_api.py",
    ]

    for pattern in search_patterns:
        expanded = Path(pattern).expanduser()
        matches = glob.glob(str(expanded))
        if matches:
            return Path(matches[0])

    return None


def patch_fast_api(fast_api_path: Path) -> bool:
    """Patch the fast_api.py file to use agent-card.json."""
    content = fast_api_path.read_text()

    # Check if using new constant (AGENT_CARD_WELL_KNOWN_PATH)
    if "AGENT_CARD_WELL_KNOWN_PATH" in content:
        print(f"[INFO] {fast_api_path} uses AGENT_CARD_WELL_KNOWN_PATH constant.")
        print("[INFO] No patching needed - already using a2a-sdk v0.3.16 standard.")
        return True

    # Check if already patched with literal string
    if "agent-card.json" in content:
        print(f"[INFO] {fast_api_path} is already patched.")
        return True

    # Find and replace the agent card URL (for older ADK versions)
    old_pattern = '/.well-known/agent.json",'
    new_pattern = '/.well-known/agent-card.json",'

    if old_pattern not in content:
        print(f"[WARNING] Could not find pattern to patch in {fast_api_path}")
        print("[WARNING] The file may have a different structure than expected.")
        print("[INFO] This may be OK if ADK is already using the new path.")
        # Check if there's any agent.json reference at all
        if "agent.json" in content and "well-known" not in content.lower():
            print("[INFO] No /.well-known/agent.json pattern found - likely already updated.")
            return True
        return False

    patched_content = content.replace(old_pattern, new_pattern)

    # Write the patched content
    fast_api_path.write_text(patched_content)
    print(f"[OK] Successfully patched {fast_api_path}")
    print(f"[OK] Changed: {old_pattern} -> {new_pattern}")
    return True


def main():
    print("=" * 60)
    print("ADK Agent Card Path Patcher")
    print("Patches Google ADK to use /.well-known/agent-card.json")
    print("=" * 60)
    print()

    fast_api_path = find_adk_fast_api()

    if fast_api_path is None:
        print("[ERROR] Could not find google-adk package in sys.path")
        print("[ERROR] Make sure google-adk is installed:")
        print("        pip install google-adk")
        sys.exit(1)

    print(f"[INFO] Found fast_api.py at: {fast_api_path}")

    success = patch_fast_api(fast_api_path)

    if success:
        print()
        print("[OK] Patching complete!")
        print("[OK] The A2A server will now serve agent cards at:")
        print("     /.well-known/agent-card.json")
        print()
        print("[NOTE] Due to a2a-sdk backward compatibility, the old path")
        print("       /.well-known/agent.json will also continue to work.")
    else:
        print()
        print("[ERROR] Patching failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
