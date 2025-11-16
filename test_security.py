#!/usr/bin/env python3
"""Test security module imports."""

import sys
sys.path.insert(0, 'secure-mediation-agent')

try:
    from security.custom_judge import secure_mediation_judge, a2a_security_judge
    print("✅ Security modules imported successfully")
    print(f"   Judge agent: {secure_mediation_judge.name}")
    print(f"   Plugin enabled: {a2a_security_judge is not None}")

    if a2a_security_judge:
        print(f"   Plugin name: {a2a_security_judge.name}")
        print(f"   Monitoring points: {len(a2a_security_judge._judge_on)}")
    else:
        print("   ⚠️  Plugin is None - dependencies missing")

except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
