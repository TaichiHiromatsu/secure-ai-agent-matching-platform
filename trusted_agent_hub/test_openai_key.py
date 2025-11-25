"""
OPENAI_API_KEY権限エラーを診断するテストスクリプト
"""
import os
import sys

def test_openai_key():
    """OPENAI_API_KEYの権限エラーを診断"""

    print("=" * 60)
    print("OPENAI_API_KEY診断テスト")
    print("=" * 60)

    # 1. 環境変数の存在確認
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"\n1. 環境変数の確認")
    print(f"   OPENAI_API_KEY exists: {api_key is not None}")

    if api_key:
        print(f"   Key length: {len(api_key)}")
        print(f"   Key prefix: {api_key[:10]}...")
        print(f"   Key suffix: ...{api_key[-4:]}")
    else:
        print("   ❌ ERROR: OPENAI_API_KEY not found in environment")
        print("\n環境変数を設定してください:")
        print("   export OPENAI_API_KEY='your-api-key'")
        return

    # 2. キー形式の検証
    print(f"\n2. キー形式の検証")
    if api_key.startswith("sk-proj-"):
        print("   ✅ Project API key detected (sk-proj-...)")
    elif api_key.startswith("sk-"):
        print("   ✅ User API key detected (sk-...)")
    else:
        print("   ⚠️  WARNING: Key doesn't start with 'sk-'")

    # 3. OpenAI API接続テスト
    print(f"\n3. OpenAI API接続テスト")
    try:
        from openai import OpenAI
        print("   ✅ openai module imported successfully")

        client = OpenAI(api_key=api_key)
        print("   ✅ OpenAI client created")

        print("   Sending test request to gpt-4o-mini...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test'"}],
            max_tokens=5
        )
        print(f"   ✅ API call successful!")
        print(f"   Response: {response.choices[0].message.content}")

    except ImportError as e:
        print(f"   ❌ Import Error: {e}")
        print("   Run: pip install openai")

    except Exception as e:
        print(f"   ❌ API Error: {type(e).__name__}")
        print(f"   Error message: {str(e)}")

        # 詳細なエラー情報
        if hasattr(e, 'status_code'):
            print(f"   Status code: {e.status_code}")
        if hasattr(e, 'response'):
            print(f"   Response: {e.response}")

    # 4. 関連する環境変数の確認
    print(f"\n4. 関連する環境変数")
    relevant_keys = ['OPENAI_API_KEY', 'OPENAI_ORG_ID', 'OPENAI_BASE_URL']
    for key in relevant_keys:
        value = os.environ.get(key)
        if value:
            if 'KEY' in key or 'TOKEN' in key:
                masked = value[:10] + "..." if len(value) > 10 else value
            else:
                masked = value
            print(f"   {key}: {masked}")
        else:
            print(f"   {key}: (not set)")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_openai_key()
