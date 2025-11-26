"""
Firebase JWT verification API for Nginx auth_request.
Returns 200 if valid, 401 if invalid.

Uses google-auth library to verify Firebase ID tokens without requiring
service account credentials.
"""
import os
from fastapi import FastAPI, Request, Response
import google.auth.transport.requests
import google.oauth2.id_token

app = FastAPI()

# Firebase project ID
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "mediation-a2a-platform")

# HTTP request object for token verification
http_request = google.auth.transport.requests.Request()


@app.get("/auth/verify")
async def verify_token(request: Request):
    """
    Verify Firebase ID token from session cookie.
    Called by Nginx auth_request directive.
    """
    session_cookie = request.cookies.get("session")

    if not session_cookie:
        print("No session cookie found")
        return Response(status_code=401)

    try:
        # Verify the ID token using Google's public keys
        # This works without service account credentials
        claims = google.oauth2.id_token.verify_firebase_token(
            session_cookie,
            http_request,
            audience=FIREBASE_PROJECT_ID
        )

        # Token is valid
        print(f"Token verified for user: {claims.get('email', claims.get('sub'))}")
        return Response(status_code=200)

    except ValueError as e:
        print(f"Token verification failed: {e}")
        return Response(status_code=401)
    except Exception as e:
        print(f"Token verification error: {e}")
        return Response(status_code=401)


@app.get("/auth/health")
async def health():
    """Health check for the auth service."""
    return {"status": "ok"}
