"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/context/AuthContext";
import { Loader2 } from "lucide-react";
import { type FormEvent, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { toast } from "sonner";

export function AuthForm() {
  const [email, setEmail] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const { login, signup } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname || "/";

  // Determine the title based on the current path
  const getTitle = () => {
    if (location.pathname === "/sign-in") {
      return "Sign In";
    } else if (location.pathname === "/sign-up") {
      return "Sign Up";
    } else {
      return "Get Started";
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await login(email, password);
      toast.success("Logged in successfully!");
      navigate(from, { replace: true });
    } catch (loginErr) {
      if (loginErr instanceof Error && loginErr.message === "Login failed") {
        try {
          await signup(email, password);
          toast.success(
            "Account created! Please check your email for confirmation.",
          );
        } catch (signupErr) {
          console.error(signupErr);
          toast.error("Signup failed. Please try again.");
        }
      } else {
        console.error(loginErr);
        toast.error("An error occurred. Please check your credentials.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center bg-muted">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle className="text-2xl font-semibold text-center">
            {getTitle()}
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <Input
              key="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email"
              className="border p-2 rounded"
              required
              disabled={isLoading}
            />
            <Input
              key="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password"
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
                  Processing...
                </>
              ) : (
                "Continue"
              )}
            </Button>
          </form>
          <p className="text-sm text-muted-foreground text-center">
            <a
              href="/auth/forgot-password"
              className="text-green-500 hover:underline"
            >
              Forgot Password?
            </a>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
