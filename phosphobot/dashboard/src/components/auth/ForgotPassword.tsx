import { useState, FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

export function ForgotPassword() {
    const [email, setEmail] = useState<string>("");
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const navigate = useNavigate();

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        setIsLoading(true);

        try {
            const response = await fetch("/auth/forgot-password", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ email }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Failed to send reset email");
            }

            toast.success("Password reset email sent! Please check your inbox.");
            setTimeout(() => navigate("/auth"), 3000);
        } catch (err) {
            toast.error(err instanceof Error ? err.message : "An error occurred");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center bg-muted">
            <Card className="w-full max-w-md">
                <CardHeader>
                    <CardTitle className="text-2xl font-semibold text-center">
                        Forgot Password
                    </CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col gap-4">
                    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                        <Input
                            type="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            placeholder="Enter your email"
                            className="border p-2 rounded"
                            required
                            disabled={isLoading}
                        />
                        <Button
                            type="submit"
                            variant="outline"
                            className="w-full cursor-pointer"
                            disabled={isLoading}
                        >
                            {isLoading ? (
                                <>
                                    <Loader2 className="size-5 mr-2 animate-spin" />
                                    Sending...
                                </>
                            ) : (
                                "Reset Password"
                            )}
                        </Button>
                    </form>
                    <p className="text-sm text-muted-foreground text-center">
                        <a href="/auth" className="text-green-500 hover:underline">
                            Back to login
                        </a>
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}