import time

from fastapi import APIRouter, HTTPException
from loguru import logger

from phosphobot.models import (
    AuthResponse,
    ConfirmRequest,
    ForgotPasswordRequest,
    LoginCredentialsRequest,
    ResetPasswordRequest,
    SessionReponse,
    StatusResponse,
)
from phosphobot.posthog import add_email_to_posthog
from phosphobot.sentry import add_email_to_sentry
from phosphobot.supabase import (
    Session,
    delete_session,
    get_client,
    save_session,
)

router = APIRouter(tags=["auth"])


@router.post("/auth/signup", response_model=SessionReponse)
async def signup(
    credentials: LoginCredentialsRequest,
) -> SessionReponse | HTTPException:
    """
    Sign up a new user.
    """
    client = await get_client()

    try:
        response = await client.auth.sign_up(
            {"email": credentials.email, "password": credentials.password}
        )
        if response.user:
            if response.session:
                # Case where email confirmation is disabled, and a session is returned
                session = Session(
                    user_id=response.user.id,
                    user_email=response.user.email,
                    email_confirmed=response.user.email_confirmed_at is not None,
                    access_token=response.session.access_token,
                    refresh_token=response.session.refresh_token,
                    expires_at=int(time.time()) + response.session.expires_in,
                )
                save_session(session)
                await client.auth.set_session(
                    access_token=response.session.access_token,
                    refresh_token=response.session.refresh_token,
                )
                add_email_to_posthog(response.user.email)
                add_email_to_sentry(response.user.email)
                return SessionReponse(message="Signup successful", session=session)
            else:
                return SessionReponse(
                    message="Signup successful, please check your email for confirmation."
                )
        else:
            # Unexpected case where user creation failed
            raise HTTPException(status_code=400, detail="Signup failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signup failed: {e}")


@router.post("/auth/signin", response_model=SessionReponse)
async def signin(
    credentials: LoginCredentialsRequest,
) -> SessionReponse | HTTPException:
    """
    Sign in an existing user.
    """
    client = await get_client()

    try:
        response = await client.auth.sign_in_with_password(
            {"email": credentials.email, "password": credentials.password}
        )
        # If signin succeeds, response.session will be present
        session = Session(
            user_id=response.user.id,
            user_email=response.user.email,
            email_confirmed=response.user.email_confirmed_at is not None,
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            expires_at=int(time.time()) + response.session.expires_in,
        )
        save_session(session)
        await client.auth.set_session(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
        )
        add_email_to_posthog(response.user.email)
        add_email_to_sentry(response.user.email)
        return SessionReponse(
            message="Signin successful",
            session=session,
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid credentials: {e}")


@router.post("/auth/logout", response_model=StatusResponse)
async def logout() -> StatusResponse | HTTPException:
    client = await get_client()

    try:
        await client.auth.sign_out()
        delete_session()
        return StatusResponse(
            message="Logout successful",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Logout failed: {e}")


@router.post("/auth/confirm", response_model=SessionReponse)
async def confirm_email(request: ConfirmRequest) -> SessionReponse | HTTPException:
    client = await get_client()

    try:
        # Set the session directly with the provided access_token and refresh_token
        response = await client.auth.set_session(
            access_token=request.access_token,
            refresh_token=request.refresh_token,
        )

        session = Session(
            user_id=response.user.id,
            user_email=response.user.email,
            email_confirmed=response.user.email_confirmed_at is not None,
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            expires_at=int(time.time()) + response.session.expires_in,
        )
        save_session(session)
        add_email_to_posthog(response.user.email)
        add_email_to_sentry(response.user.email)

        return SessionReponse(
            message="Email confirmed successfully",
            session=session,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid or expired token: {str(e)}"
        )


@router.get("/auth/check_auth", response_model=AuthResponse)
async def is_authenticated() -> AuthResponse:
    """
    Check if the user is authenticated by validating the session with Supabase.
    Returns a JSON response indicating authentication status.
    """

    try:
        client = await get_client()
        session = await client.auth.get_session()
        if session:
            return AuthResponse(
                authenticated=True,
                session=Session(
                    user_id=session.user.id,
                    user_email=session.user.email,
                    email_confirmed=session.user.email_confirmed_at is not None,
                    access_token=session.access_token,
                    refresh_token=session.refresh_token,
                    expires_at=int(time.time()) + session.expires_in,
                ),
            )
        else:
            return AuthResponse(authenticated=False)
    except Exception as e:
        # Log unexpected errors but don’t expose them to the client
        logger.warning(f"Error checking authentication: {e}")
        return AuthResponse(authenticated=False)


@router.post("/auth/forgot-password", response_model=StatusResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
) -> StatusResponse | HTTPException:
    """
    Send a password reset email to the provided email address.
    """
    client = await get_client()

    try:
        await client.auth.reset_password_for_email(
            email=request.email,
        )
        logger.info(f"Password reset email sent to: {request.email}")
        return StatusResponse(message="Password reset email sent successfully.")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to send reset email: {str(e)}"
        )


@router.post("/auth/reset-password", response_model=StatusResponse)
async def reset_password(
    request: ResetPasswordRequest,
) -> StatusResponse | HTTPException:
    """
    Reset a user's password using the recovery tokens from the Supabase reset email.
    """
    client = await get_client()

    logger.info(f"Received reset request with access_token: {request.access_token}")
    try:
        # Authenticate the session with the provided tokens
        response = await client.auth.set_session(
            access_token=request.access_token,
            refresh_token=request.refresh_token,
        )
        logger.info(f"Session set: user={response.user.id}")

        # Update the user's password
        await client.auth.update_user({"password": request.new_password})
        logger.info(f"Password updated for user: {response.user.id}")

        # Optionally, update the session (though not strictly necessary for reset)
        session = Session(
            user_id=response.user.id,
            user_email=response.user.email,
            email_confirmed=response.user.email_confirmed_at is not None,
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            expires_at=int(time.time()) + response.session.expires_in,
        )
        save_session(session)
        return StatusResponse(
            message="Password reset successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid or expired token: {str(e)}"
        )
