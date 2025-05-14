import { useEffect } from "react";
import { toast } from "sonner";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import { Session } from "@/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function ConfirmEmail() {
    const { login } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        const confirmEmail = async () => {
            const hash = window.location.hash.substring(1);
            const params = new URLSearchParams(hash);
            const accessToken = params.get("access_token");
            const refreshToken = params.get("refresh_token");

            if (!accessToken || !refreshToken) {
                toast.error("Missing access_token or refresh_token in URL");
                navigate("/");
                return;
            }

            try {
                const response = await fetch("/auth/confirm", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        access_token: accessToken,
                        refresh_token: refreshToken,
                    }),
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || "Email confirmation failed");
                }

                const data: { message: string; session: Session } = await response.json();
                console.log("Confirmation data:", data);

                localStorage.setItem("session", JSON.stringify(data.session));
                login(data.session.user_email, "", data.session);
                toast.success("Email confirmed successfully!");
                navigate("/");
            } catch (err) {
                toast.error(err instanceof Error ? err.message : "An error occurred");
            }
        };

        confirmEmail();
    }, [login, navigate]);

    return (
        <div className="flex items-center justify-center bg-muted">
            <Card className="w-full max-w-md">
                <CardHeader>
                    <CardTitle className="text-2xl font-semibold text-center">
                        Email Confirmation
                    </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col gap-4">
                    <p className="text-center text-muted-foreground">Confirming your email...</p>
                </CardContent>
            </Card>
        </div>
    );
}